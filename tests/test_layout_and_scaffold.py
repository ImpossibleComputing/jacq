import tempfile
import unittest
from pathlib import Path

from integrations.vllm_kivi.layout import KiviModelGeometry, KiviPageLayout
from integrations.vllm_kivi.source_patch import (
    ADDED_FILES,
    _patch_sched_output,
    apply_scaffold,
    build_unified_diff,
    inspect_checkout,
)


def _write_minimal_vllm_tree(root: Path) -> None:
    files = {
        "CMakeLists.txt": '''set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/merge_attn_states.cu"
  "csrc/pos_encoding_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")
endif()
''',
        "vllm/config.py": '''CacheDType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]\n\n        elif self.cache_dtype in get_args(CacheDType):\n            logger.info(\n                "Using fp8 data type to store kv cache. It reduces the GPU "\n                "memory footprint and boosts the performance. "\n                "Meanwhile, it may cause accuracy drop without a proper "\n                "scaling factor")\n        else:\n            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")\n\n\n@dataclass\nclass VllmConfig:\n    def __post_init__(self):\n        """Verify configs are valid & consistent with each other.\n        """\n        pass\n''',
        "vllm/utils.py": '''STR_DTYPE_TO_TORCH_DTYPE = {\n    "fp8_e5m2": torch.uint8,\n    "int8": torch.int8,\n}\n''',
        "vllm/v1/kv_cache_interface.py": '''class FullAttentionSpec(AttentionSpec):\n\n    @property\n    def type_id(self) -> str:\n        return f"full_attention_{self.block_size}_{self.page_size_bytes}"\n\n    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:\n        max_model_len = vllm_config.model_config.max_model_len\n        return cdiv(max_model_len, self.block_size) * self.page_size_bytes\n\n\n@dataclass\nclass SlidingWindowSpec(AttentionSpec):\n    pass\n''',
        "vllm/v1/attention/backends/flash_attn.py": '''from vllm import _custom_ops as ops

    local_attn_metadata: Optional[LocalAttentionMetadata] = None

            local_attn_metadata=local_attn_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata

        if is_quantized_kv_cache(self.kv_cache_dtype) \\
            and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

        num_actual_tokens = attn_metadata.num_actual_tokens
        # Reshape the input keys and values and store them in the cache.
        key_cache, value_cache = kv_cache.unbind(0)
''',
        "vllm/engine/arg_utils.py": '''        # No Fp8 KV cache so far.
        if self.kv_cache_dtype != "auto":
            fp8_attention = self.kv_cache_dtype.startswith("fp8")
            will_use_fa = (
                current_platform.is_cuda()
                and not envs.is_set("VLLM_ATTENTION_BACKEND")
            ) or envs.VLLM_ATTENTION_BACKEND == "FLASH_ATTN_VLLM_V1"
            supported = False
            if fp8_attention and will_use_fa:
                from vllm.attention.utils.fa_utils import (
                    flash_attn_supports_fp8)
                supported = flash_attn_supports_fp8()
            if not supported:
                _raise_or_fallback(feature_name="--kv-cache-dtype",
                                   recommend_to_remove=False)
                return False
''',
        "vllm/v1/core/specialized_manager.py": '''from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheSpec,
                                        SlidingWindowSpec)

spec_manager_map: dict[type[KVCacheSpec], type[SpecializedManager]] = {
    FullAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
}
''',
        "vllm/v1/core/kv_cache_manager.py": '''from vllm.v1.core.block_pool import BlockPool

class KVCacheManager:

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        When caching is enabled, we free the blocks in reverse order so that
        the tail blocks are evicted first.
        """
        blocks = self.req_to_blocks.pop(request.request_id, [])
        self.block_pool.free_blocks(blocks)
''',
        "vllm/v1/core/sched/output.py": 'from dataclasses import dataclass\nfrom typing import TYPE_CHECKING, Optional\n\n@dataclass\nclass NewRequestData:\n    req_id: str\n    block_ids: list[int]\n    num_computed_tokens: int\n    lora_request: Optional[LoRARequest]\n\n@dataclass\nclass CachedRequestData:\n    req_id: str\n    resumed_from_preemption: bool\n    new_token_ids: list[int]\n    new_block_ids: list[int]\n    num_computed_tokens: int\n\n@dataclass\nclass SchedulerOutput:\n    scheduled_new_reqs: list[NewRequestData]\n    scheduled_cached_reqs: list[CachedRequestData]\n',
        "vllm/v1/core/sched/scheduler.py": '''import time

from vllm.v1.core.kv_cache_manager import KVCacheManager

class Scheduler:

    def __init__(self):
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats)

    def schedule(self):
        req_to_new_block_ids: dict[str, list[int]] = {}
        num_scheduled_tokens: dict[str, int] = {}

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]

        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]

    def _make_cached_request_data(self):
        return req_data

    def _try_schedule_encoder_inputs(
        self,
    ):
        pass
''',
        "vllm/v1/worker/gpu_model_runner.py": '''from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,\n                                        KVCacheConfig, KVCacheSpec,\n                                        SlidingWindowSpec)\n\n        self._update_states(scheduler_output)\n        if not scheduler_output.total_num_scheduled_tokens:\n\n                if isinstance(kv_cache_spec, AttentionSpec):\n                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(\n                        num_blocks, kv_cache_spec.block_size,\n                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)\n                    dtype = kv_cache_spec.dtype\n                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,\n                                                        dtype=dtype,\n                                                        device=self.device)\n                else:\n                    # TODO: add new branches when introducing more types of\n                    # KV cache specs.\n                    raise ValueError("Unknown KV cache spec type.")\n\n                    kv_cache_spec[layer_name] = FullAttentionSpec(\n                        block_size=block_size,\n                        num_kv_heads=attn_module.num_kv_heads,\n                        head_size=attn_module.head_size,\n                        dtype=self.kv_cache_dtype,\n                        use_mla=use_mla)\n\n            self.input_batch.block_table.append_row(req_data.new_block_ids,\n                                                    req_index)\n''',
        "csrc/ops.h": '''void paged_attention_v1(\n''',
        "csrc/torch_bindings.cpp": '''  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);\n''',
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


class KiviVllmLayoutTest(unittest.TestCase):
    def test_checked_in_patch_matches_added_scaffold_files(self):
        patch = Path(
            "integrations/vllm_kivi/patches/vllm-0.8.5-int4-kivi-scaffold.patch"
        ).read_text()
        for path, source in ADDED_FILES.items():
            with self.subTest(path=path):
                section = patch.split(f"+++ b/{path}\n", 1)[1]
                lines = section.splitlines(keepends=True)
                self.assertTrue(lines[0].startswith("@@ "))
                added = []
                for line in lines[1:]:
                    if line.startswith(("--- ", "diff --git ")):
                        break
                    if line.startswith("+"):
                        added.append(line[1:])
                    elif not line.startswith("@@ "):
                        self.fail(f"unexpected patch line for {path}: {line[:40]!r}")
                self.assertEqual("".join(added), source)

    def test_qwen32b_kivi_vllm_page_abi_matches_capacity_ceiling(self):
        layout = KiviPageLayout(KiviModelGeometry.qwen25_coder_32b())
        abi = layout.to_abi_dict(seq_len=4096)
        self.assertEqual(abi["fp16_page_bytes_per_layer"], 131_072)
        self.assertEqual(abi["key_code_bytes"], 16_384)
        self.assertEqual(abi["value_code_bytes"], 16_384)
        self.assertEqual(abi["key_scale_zero_bytes"], 4_096)
        self.assertEqual(abi["value_scale_zero_bytes"], 4_096)
        self.assertEqual(abi["quantized_page_bytes_per_layer"], 40_960)
        self.assertEqual(abi["residual_pages"], 4)
        self.assertEqual(abi["effective_bytes_per_token_all_layers_at_seq_len"], 87_552.0)
        self.assertGreater(abi["capacity_ratio_vs_fp16_at_seq_len"], 2.99)
        self.assertLess(abi["capacity_ratio_vs_fp16_at_seq_len"], 3.0)

    def test_layout_rejects_block16_because_key_groups_straddle_pages(self):
        layout = KiviPageLayout(KiviModelGeometry.qwen25_coder_32b(), block_size=16)
        with self.assertRaisesRegex(ValueError, "block_size==group_size"):
            layout.validate_for_vllm_lane()

    def test_sched_output_sparse_metadata_targets_cached_request_data_only(self):
        text = (
            "from dataclasses import dataclass\n"
            "from typing import TYPE_CHECKING, Optional\n\n"
            "@dataclass\n"
            "class NewRequestData:\n"
            "    req_id: str\n"
            "    block_ids: list[int]\n"
            "    num_computed_tokens: int\n"
            "    lora_request: Optional[LoRARequest]\n\n"
            "@dataclass\n"
            "class CachedRequestData:\n"
            "    req_id: str\n"
            "    resumed_from_preemption: bool\n"
            "    new_token_ids: list[int]\n"
            "    new_block_ids: list[int]\n"
            "    num_computed_tokens: int\n"
        )
        patched = _patch_sched_output(text)
        self.assertIn("from typing import Any, TYPE_CHECKING, Optional", patched)
        new_block = patched.split("@dataclass\nclass CachedRequestData:", 1)[0]
        cached_block = patched.split("@dataclass\nclass CachedRequestData:", 1)[1]
        self.assertNotIn("kivi_sparse_metadata", new_block)
        self.assertIn("kivi_sparse_metadata: Optional[dict[str, Any]] = None", cached_block)
        self.assertLess(cached_block.index("num_computed_tokens: int"), cached_block.index("kivi_sparse_metadata"))

    def test_backend_stub_cache_writer_selector_fails_closed(self):
        # The selector lives inside the generated vLLM backend stub, not as an
        # importable jacq-side helper. Execute the stub with a tiny fake torch
        # module so this remains a CPU/no-vLLM unit test.
        import os
        import sys
        import types

        from integrations.vllm_kivi import source_patch

        sentinel = object()
        old_torch = sys.modules.get("torch", sentinel)
        old_env = os.environ.get("JAC_KIVI_CACHE_WRITE_KERNEL", sentinel)
        sys.modules["torch"] = types.SimpleNamespace()
        ns: dict[str, object] = {}
        try:
            exec(source_patch.KIVI_BACKEND_STUB, ns)
            os.environ.pop("JAC_KIVI_CACHE_WRITE_KERNEL", None)
            self.assertEqual(ns["cache_write_kernel_source"](), "hand_cuda")
            os.environ["JAC_KIVI_CACHE_WRITE_KERNEL"] = "jac-dsl"
            self.assertEqual(ns["cache_write_kernel_source"](), "jac_dsl")
            os.environ["JAC_KIVI_CACHE_WRITE_KERNEL"] = "bogus"
            with self.assertRaisesRegex(ValueError, "JAC_KIVI_CACHE_WRITE_KERNEL"):
                ns["cache_write_kernel_source"]()
        finally:
            if old_torch is sentinel:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch
            if old_env is sentinel:
                os.environ.pop("JAC_KIVI_CACHE_WRITE_KERNEL", None)
            else:
                os.environ["JAC_KIVI_CACHE_WRITE_KERNEL"] = old_env

    def test_source_patch_scaffold_anchors_and_diff_are_actionable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_minimal_vllm_tree(root)
            report = inspect_checkout(root)
            self.assertTrue(report["ok_to_apply_scaffold"])
            self.assertEqual(report["layout_abi"]["quantized_page_bytes_per_layer"], 40_960)
            diff = build_unified_diff(root)
            self.assertIn("int4_kivi", diff)
            self.assertIn("KiviFullAttentionSpec", diff)
            self.assertIn("csrc/kivi/decode_attention_kivi_dsl.cu", diff)
            self.assertIn("int4_kivi_dsl_paged", diff)
            self.assertIn("K_CODE_PAGES", diff)
            self.assertIn("jac_launch_order", diff)
            # Page storage (K/V_CODE_PAGES etc.) must be sized to vLLM's
            # FULL, real physical block count, not reduced by residual_pages:
            # BLOCK_TABLE entries are assigned from the full pool regardless
            # of which logical page they back, so a smaller allocation would
            # leave the withheld top block ids unbacked and an aged-out page
            # would write out of bounds. See kivi_cache_tensor_shapes's
            # docstring in source_patch.py for the full account.
            self.assertIn(
                '"K_CODE_PAGES": (num_blocks, self.num_kv_heads, self.block_size,',
                diff)
            self.assertNotIn("old_num_blocks", diff)
            self.assertNotIn('"old_blocks": old_blocks', diff)
            self.assertIn("vllm/v1/attention/backends/kivi_dsl.py", diff)
            self.assertIn("vllm/v1/core/kivi_page_aware_allocator.py", diff)
            self.assertIn("apply_kivi_page_aware_drop", diff)
            self.assertIn("apply_page_aware_drop_to_vllm_blocks", diff)
            self.assertIn("csrc/kivi/decode_attention_kivi_binding.cpp", diff)
            self.assertIn("csrc/kivi/cache_write_kivi_dsl.cu", diff)
            self.assertIn("extern \"C\" void jac_launch", diff)
            self.assertIn("kivi_cache_write_launch", diff)
            # batched prefill-loop fix: a batched sibling of kivi_cache_write_launch that appends
            # NUM_NEW new tokens in one launch instead of NUM_NEW sequential single-token launches
            # (see plan_batched_write in the scaffolded kivi_dsl.py for the sub-chunk scheduling this
            # pairs with). Source-level addition only as of this landing -- not yet wired into the
            # request loop below, not yet GPU-bitchecked against repeated single-token calls.
            self.assertIn("kivi_cache_write_batched_launch", diff)
            self.assertIn("append_scratch_tokens_batched_kernel", diff)
            # torch op bindings for both batched kernels (batched): registered alongside the
            # existing single-token ops, not replacing them -- both paths coexist until the opt-in
            # flag decides which one a given request loop iteration uses.
            self.assertIn("kivi_decode_batched_dsl", diff)
            self.assertIn("kivi_cache_write_batched_dsl", diff)
            self.assertIn("jac_launch_batched", diff)
            self.assertIn("csrc/kivi/decode_attention_kivi_batched_dsl.cu", diff)
            self.assertIn("jac_launch_sparse_retained", diff)
            self.assertIn("kivi_decode_sparse_retained_dsl", diff)
            self.assertIn("csrc/kivi/decode_attention_kivi_sparse_retained_dsl.cu", diff)
            # H100 source-build gate (found live): with PyTorch 2.6, torch::kCUDA
            # registered these vLLM custom ops under CPU, so the CUDA KIVI
            # bitchecks failed before any serving measurement.  The scaffold
            # must pin the CUDA binding TU to explicit DispatchKey values.
            self.assertIn("c10::DispatchKey::CUDA", diff)
            self.assertIn(
                'ops.impl("kivi_cache_write_dsl", c10::DispatchKey::CUDA',
                diff)
            # Request-loop wiring (batched): opt-in flag, default OFF -- the existing per-token
            # path (checked further below) must remain reachable and unconditional-by-default.
            self.assertIn('os.environ.get("JAC_KIVI_BATCHED_PREFILL") == "1"', diff)
            self.assertIn("plan_batched_write(start_seq_len, num_tok_this_req)", diff)
            self.assertIn("require_kivi_cache_write_batched(", diff)
            self.assertIn("launch_kivi_dsl_decode_batched(", diff)
            # A sub-chunk that ages out a page must not batch decode over it:
            # the age-out shifts residual scratch at the sub-chunk's own last
            # token, so earlier tokens must decode before the shift. The
            # scaffold splits an aging-out sub-chunk into "every token except
            # the last" (batched, no shift yet) + "the last token alone"
            # (decodes only after its own write/shift).
            self.assertIn("age_out_page is not None and sub_chunk.num_new > 1", diff)
            self.assertIn("kivi_decode_dsl", diff)
            self.assertIn("static constexpr int NH = 40", diff)
            self.assertIn("TP2_LOCAL_NH = 20", diff)
            self.assertIn("TP2_LOCAL_NKV = 4", diff)
            self.assertIn("check_supported_geometry(runtime_nh, runtime_nkv", diff)
            self.assertIn("check_shape(q, {runtime_nh, HD}", diff)
            self.assertIn("jac_launch_tp2_local", diff)
            self.assertIn("decode_attention_kivi_tp2_local_dsl.cu", diff)
            self.assertIn("launch_kivi_dsl_decode", diff)
            self.assertIn("require_kivi_cache_write", diff)
            self.assertIn("pack_key_age_out_page_kernel", diff)
            self.assertIn("with torch.cuda.stream(stream)", diff)
            self.assertIn("torch.ops._C.kivi_decode_dsl", diff)
            self.assertIn("launch_kivi_dsl_decode", diff)
            # follow-on: the single-token writer can be explicitly
            # routed to the DSL-authored op, but the incumbent hand-CUDA writer
            # remains the default until a served parity/default-flip gate lands.
            self.assertIn('CACHE_WRITE_KERNEL_ENV = "JAC_KIVI_CACHE_WRITE_KERNEL"', diff)
            self.assertIn("cache_write_kernel_source", diff)
            self.assertIn("kivi_cache_write_jac_dsl", diff)
            self.assertIn("kivi_cache_write_dsl", diff)
            # The loop must iterate requests (num_reqs =
            # attn_metadata.seq_lens.shape[0]), NOT total scheduled tokens
            # (num_actual_tokens): these only coincide when every request
            # contributes exactly one token (pure decode); a prefill chunk
            # makes num_actual_tokens exceed num_reqs. Token rows map to
            # requests via attn_metadata.query_start_loc, which is exactly
            # what the scaffold uses instead of assuming or rejecting the
            # mismatch.
            self.assertIn("num_reqs = attn_metadata.seq_lens.shape[0]", diff)
            self.assertIn("query_start_loc_list = attn_metadata.query_start_loc.tolist()", diff)
            self.assertIn("more than max_num_reqs={max_seqs} residual/scratch", diff)
            self.assertIn("for j in range(num_reqs):", diff)
            self.assertIn("for local_tok in range(num_tok_this_req):", diff)
            self.assertIn("output[:n].copy_", diff)
            self.assertIn("sparse_retained_metadata_by_req_id", diff)
            self.assertIn("kivi_sparse_metadata", diff)
            self.assertIn("check_block_table_width", diff)
            checked_patch = Path("integrations/vllm_kivi/patches/vllm-0.8.5-int4-kivi-scaffold.patch").read_text()
            self.assertIn("vllm/v1/attention/backends/kivi_dsl.py", checked_patch)
            self.assertIn("vllm/v1/core/kivi_page_aware_allocator.py", checked_patch)
            self.assertIn("apply_kivi_page_aware_drop", checked_patch)
            self.assertIn("csrc/kivi/decode_attention_kivi_binding.cpp", checked_patch)
            self.assertIn("csrc/kivi/cache_write_kivi_dsl.cu", checked_patch)
            self.assertIn("launch_kivi_dsl_decode", checked_patch)
            self.assertIn("require_kivi_cache_write", checked_patch)
            self.assertIn("kivi_cache_write_dsl", checked_patch)
            self.assertIn("K_RESIDUAL_SCRATCH", checked_patch)
            self.assertIn("kivi_cache_write_batched_launch", checked_patch)
            self.assertIn("append_scratch_tokens_batched_kernel", checked_patch)
            self.assertIn("kivi_decode_batched_dsl", checked_patch)
            self.assertIn("kivi_cache_write_batched_dsl", checked_patch)
            self.assertIn("jac_launch_batched", checked_patch)
            self.assertIn("csrc/kivi/decode_attention_kivi_batched_dsl.cu", checked_patch)
            self.assertIn("jac_launch_sparse_retained", checked_patch)
            self.assertIn("kivi_decode_sparse_retained_dsl", checked_patch)
            self.assertIn("csrc/kivi/decode_attention_kivi_sparse_retained_dsl.cu", checked_patch)
            self.assertIn('os.environ.get("JAC_KIVI_BATCHED_PREFILL") == "1"', checked_patch)
            self.assertIn("plan_batched_write(start_seq_len, num_tok_this_req)", checked_patch)
            self.assertIn("age_out_page is not None and sub_chunk.num_new > 1", checked_patch)
            self.assertIn('CACHE_WRITE_KERNEL_ENV = "JAC_KIVI_CACHE_WRITE_KERNEL"', checked_patch)
            self.assertIn("cache_write_kernel_source", checked_patch)
            # the DSL-authored writer, comparison-only (not wired into the
            # request loop) -- a genuinely separate destination path from
            # cache_write_kivi_dsl.cu above (that one is hand-CUDA despite
            # its name).
            self.assertIn("jac_launch_cache_write", diff)
            self.assertIn("kivi_cache_write_jac_dsl", diff)
            self.assertIn("csrc/kivi/cache_write_kivi_jac_dsl.cu", diff)
            self.assertIn("jac_launch_cache_write", checked_patch)
            self.assertIn("kivi_cache_write_jac_dsl", checked_patch)
            self.assertIn("csrc/kivi/cache_write_kivi_jac_dsl.cu", checked_patch)
            # Batched DSL writer comparison artifact: also not request-loop wired until GPU exact.
            self.assertIn("jac_launch_cache_write_batched", diff)
            self.assertIn("kivi_cache_write_batched_jac_dsl", diff)
            self.assertIn("csrc/kivi/cache_write_kivi_batched_jac_dsl.cu", diff)
            self.assertIn("jac_launch_cache_write_batched", checked_patch)
            self.assertIn("kivi_cache_write_batched_jac_dsl", checked_patch)
            self.assertIn("csrc/kivi/cache_write_kivi_batched_jac_dsl.cu", checked_patch)
            # Writer and decode both use the SAME real vLLM block
            # table again (the wrapper's validation enforces the full
            # invariant); the private per-bundle identity
            # workaround is gone. Check the exact allocation
            # snippet, not a bare substring -- explanatory prose in the
            # source is allowed to mention the old name by way of history.
            self.assertIn("block_table = attn_metadata.block_table[j]", checked_patch)
            self.assertIn("check_block_table_width", checked_patch)
            self.assertNotIn('"BLOCK_TABLE_IDENTITY": torch.arange(', checked_patch)
            self.assertNotIn('block_table=kv_cache["BLOCK_TABLE_IDENTITY"]', checked_patch)
            apply_scaffold(root)
            config = (root / "vllm/config.py").read_text()
            self.assertIn("def _validate_int4_kivi_compatibility(", config)
            self.assertIn("_validate_int4_kivi_compatibility(\n"
                          "            self.model_config, self.cache_config, "
                          "self.parallel_config)", config)
            generated = root / "vllm/v1/attention/backends/kivi_dsl.py"
            self.assertTrue(generated.exists())
            self.assertIn("KiviDslLaunchBinding", generated.read_text())
            self.assertIn("require_kivi_cache_write", generated.read_text())
            self.assertIn("cache_write_kernel_source", generated.read_text())
            self.assertIn("kivi_cache_write_jac_dsl", generated.read_text())
            self.assertIn("kivi_cache_write_dsl", generated.read_text())
            binding = root / "csrc/kivi/decode_attention_kivi_binding.cpp"
            self.assertTrue(binding.exists())
            self.assertIn("jac_launch", binding.read_text())
            self.assertIn("jac_launch_sparse_retained", binding.read_text())
            self.assertIn("kivi_decode_sparse_retained_dsl", binding.read_text())
            self.assertIn("check_shape(k_code_pages", binding.read_text())
            self.assertIn("kivi_cache_write_dsl", binding.read_text())
            self.assertIn("kivi_cache_write_batched_jac_dsl", binding.read_text())
            self.assertIn("K_RESIDUAL_SCRATCH", binding.read_text())
            writer = root / "csrc/kivi/cache_write_kivi_dsl.cu"
            self.assertTrue(writer.exists())
            self.assertIn("pack_value_age_out_page_kernel", writer.read_text())
            flash = root / "vllm/v1/attention/backends/flash_attn.py"
            self.assertIn("launch_kivi_dsl_decode", flash.read_text())
            self.assertIn("require_kivi_cache_write", flash.read_text())
            self.assertIn("int4_kivi_dsl_paged", flash.read_text())
            kvm = root / "vllm/v1/core/kv_cache_manager.py"
            self.assertIn("apply_kivi_page_aware_drop", kvm.read_text())
            allocator = root / "vllm/v1/core/kivi_page_aware_allocator.py"
            self.assertTrue(allocator.exists())
            self.assertIn("BlockPool", allocator.read_text())


if __name__ == "__main__":
    unittest.main()
