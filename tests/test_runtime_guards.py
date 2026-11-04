"""CPU source/scaffold tests for the fail-closed runtime guards emitted into
the vLLM scaffold, plus executable checks of the Python-side guards.

These tests verify that the emitted C++ torch-op boundary
(KIVI_BINDING_SOURCE) and the materialized backend stub
(KIVI_BACKEND_STUB) contain and invoke the guards. They do NOT compile or
execute any CUDA: no GPU behavior, kernel correctness, or serving claim is
made or implied here (see docs/LIMITATIONS.md).
"""
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from integrations.vllm_kivi import preflight, source_patch  # noqa: E402

BINDING = source_patch.KIVI_BINDING_SOURCE
STUB = source_patch.KIVI_BACKEND_STUB

# Every torch-op launch path defined in KIVI_BINDING_SOURCE. Each one must
# invoke check_launch_device over its full tensor set before launching.
LAUNCH_OP_NAMES = (
    "kivi_decode_dsl",
    "kivi_decode_sparse_retained_dsl",
    "kivi_decode_batched_dsl",
    "kivi_cache_write_dsl",
    "kivi_cache_write_jac_dsl",
    "kivi_cache_write_batched_dsl",
    "kivi_cache_write_batched_jac_dsl",
)


def _op_body(op_name: str) -> str:
    """Extract the body of `void <op_name>(...) { ... }` from the C++ source."""
    pattern = re.compile(
        r"void " + re.escape(op_name) + r"\(.*?\n}\n", re.DOTALL
    )
    match = pattern.search(BINDING)
    if match is None:
        raise AssertionError(f"op {op_name} not found in KIVI_BINDING_SOURCE")
    return match.group(0)


class BindingDeviceGuardSourceTest(unittest.TestCase):
    """Source-level checks on the emitted C++ device/arch guard."""

    def test_guard_helper_exists_and_reads_actual_device(self):
        self.assertIn("static void check_launch_device(", BINDING)
        # the guard must query the runtime device, not a build-time constant
        self.assertIn("cudaGetDeviceProperties(&prop, device.index())", BINDING)
        # fail-closed on the cudaGetDeviceProperties call itself
        self.assertIn("prop_err == cudaSuccess", BINDING)
        # accept exactly the validated compute capabilities
        self.assertIn("compute_cap == 80 || compute_cap == 89", BINDING)
        self.assertIn("outside the validated set sm_80/sm_89", BINDING)

    def test_guard_rejects_mixed_devices(self):
        self.assertIn("must all live on one CUDA device", BINDING)
        self.assertIn("t->device() == device", BINDING)

    def test_every_launch_path_calls_the_device_guard(self):
        missing = [
            op for op in LAUNCH_OP_NAMES if "check_launch_device(" not in _op_body(op)
        ]
        self.assertEqual(missing, [], "launch paths missing check_launch_device")

    def test_guard_call_names_its_op(self):
        # the guard error must identify which op failed
        for op in LAUNCH_OP_NAMES:
            self.assertIn(f'"{op}"', _op_body(op))

    def test_decode_guard_covers_all_tensors(self):
        # spot-check the representative per-op tensor sets: every tensor
        # parameter of the op must be passed by pointer into the guard
        # (the last list element is followed by '}' rather than ',')
        decode_body = _op_body("kivi_decode_dsl")
        for name in (
            "q", "block_table", "k_code_pages", "k_scale_pages",
            "k_zero_pages", "v_code_pages", "v_scale_pages", "v_zero_pages",
            "k_residual_scratch", "v_residual_scratch", "out",
        ):
            self.assertTrue(
                f"&{name}," in decode_body or f"&{name}}}" in decode_body,
                f"kivi_decode_dsl guard must cover {name}")
        batched_body = _op_body("kivi_decode_batched_dsl")
        for name in ("q", "seq_lens", "block_table", "out"):
            self.assertTrue(
                f"&{name}," in batched_body or f"&{name}}}" in batched_body)
        write_body = _op_body("kivi_cache_write_dsl")
        for name in ("k_code_pages", "new_k", "new_v", "slot_mapping",
                     "block_table"):
            self.assertTrue(
                f"&{name}," in write_body or f"&{name}}}" in write_body)


class BindingBlockTableWidthSourceTest(unittest.TestCase):
    """The emitted boundary must accept shorter configured context windows."""

    def test_width_uses_active_sequence_length_in_every_dense_launch(self):
        self.assertIn(
            "seq_len > RESIDUAL ? (seq_len - RESIDUAL) / BLOCK_SIZE : 0;",
            BINDING,
        )
        self.assertNotIn("t.size(0) >= OLD_BLOCKS", BINDING)
        single = ("kivi_decode_dsl", "kivi_cache_write_dsl", "kivi_cache_write_jac_dsl")
        batched_write = ("kivi_cache_write_batched_dsl", "kivi_cache_write_batched_jac_dsl")
        for op in single:
            self.assertIn(
                'check_block_table_width(block_table, "BLOCK_TABLE", seq_len);',
                _op_body(op), op,
            )
        for op in batched_write:
            self.assertIn(
                'check_block_table_width(block_table, "BLOCK_TABLE", start_seq_len + num_new);',
                _op_body(op), op,
            )
        batched_decode = _op_body("kivi_decode_batched_dsl")
        self.assertIn(
            'check_block_table_width(block_table, "BLOCK_TABLE", seq_lens_values[i]);',
            batched_decode,
        )
        self.assertLess(
            batched_decode.index('"SEQ_LENS["'),
            batched_decode.index('check_block_table_width(block_table'),
        )
        self.assertLess(
            batched_decode.index('check_block_table_width(block_table'),
            batched_decode.index('check_launch_device('),
        )

    def test_emitted_cpp_guard_at_old_page_boundaries(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        start = BINDING.index("static void check_block_table_width(")
        end = BINDING.index("\n// Fail-closed actual-device guard", start)
        helper = BINDING[start:end]
        program = r'''
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
namespace torch {
struct Tensor {
  int64_t width;
  int64_t dim() const { return 1; }
  int64_t size(int) const { return width; }
};
}
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("guard"); } while (false)
static constexpr int OLD_BLOCKS = 124;
static constexpr int BLOCK_SIZE = 32;
static constexpr int RESIDUAL = 128;
''' + helper + r'''
int main() {
  const int64_t cases[][2] = {
      {128, 0}, {129, 0}, {159, 0}, {160, 1},
      {191, 1}, {192, 2}, {4096, 124}};
  for (const auto& item : cases) {
    const int64_t seq_len = item[0];
    const int64_t required = item[1];
    check_block_table_width(torch::Tensor{required}, "BLOCK_TABLE", seq_len);
    if (required > 0) {
      bool rejected = false;
      try {
        check_block_table_width(torch::Tensor{required - 1}, "BLOCK_TABLE", seq_len);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      if (!rejected) return 1;
    }
  }
  for (int64_t invalid : {int64_t{0}, int64_t{4097}}) {
    bool rejected = false;
    try {
      check_block_table_width(torch::Tensor{124}, "BLOCK_TABLE", invalid);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected) return 2;
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "guard.cpp"
            binary = Path(directory) / "guard"
            source.write_text(program)
            compile_result = subprocess.run(
                [compiler, "-std=c++17", str(source), "-o", str(binary)],
                capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)


class BindingSeqLensValueGuardSourceTest(unittest.TestCase):
    """Source-level checks on the batched-decode SEQ_LENS value guard."""

    def test_batched_decode_checks_seq_lens_values_before_launch(self):
        body = _op_body("kivi_decode_batched_dsl")
        self.assertIn("seq_lens.to(at::kCPU)", body)
        self.assertIn("SEQ_LENS[", body)
        self.assertIn("[1, 4096]", body)
        # the guard must run BEFORE the launch call
        self.assertLess(body.index("seq_lens.to(at::kCPU)"),
                        body.index("jac_launch_batched_tp2_local"))

    def test_batched_decode_guard_discloses_host_sync_cost(self):
        body = _op_body("kivi_decode_batched_dsl")
        self.assertIn("device-to-host sync", body)
        self.assertIn("JAC_KIVI_BATCHED_PREFILL", body)

    def test_checked_in_patch_carries_both_guards(self):
        patch = REPO_ROOT.joinpath(
            "integrations/vllm_kivi/patches/vllm-0.8.5-int4-kivi-scaffold.patch"
        ).read_text()
        self.assertIn("static void check_launch_device(", patch)
        self.assertIn("compute_cap == 80 || compute_cap == 89", patch)
        self.assertIn("seq_lens.to(at::kCPU)", patch)
        self.assertIn("SEQ_LENS[", patch)
        self.assertIn(
            "seq_len > RESIDUAL ? (seq_len - RESIDUAL) / BLOCK_SIZE : 0;",
            patch,
        )
        self.assertNotIn("t.size(0) >= OLD_BLOCKS", patch)


class BackendStubSeqLensGuardTest(unittest.TestCase):
    """Executable check of the Python serving wrapper's SEQ_LENS guard.

    The stub is executed with a minimal fake torch module (the same pattern
    as the cache-writer selector test); this stays a CPU-only unit test and
    executes no CUDA.
    """

    def _exec_stub(self):
        recorded = []

        class _FakeTensor:
            def __init__(self, dtype, device="cuda:0", values=()):
                self.dtype = dtype
                self.device = device
                self._values = list(values)

            def contiguous(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self._values)

        int32, int64, uint8, fp16, fp32 = (
            object(), object(), object(), object(), object())
        fake_torch = types.SimpleNamespace(
            int32=int32,
            int64=int64,
            uint8=uint8,
            float16=fp16,
            float32=fp32,
            ops=types.SimpleNamespace(
                _C=types.SimpleNamespace(
                    kivi_decode_batched_dsl=lambda *a: recorded.append(a),
                )
            ),
        )
        ns = {}
        sentinel = object()
        old_torch = sys.modules.get("torch", sentinel)
        sys.modules["torch"] = fake_torch
        try:
            exec(STUB, ns)
        finally:
            if old_torch is sentinel:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch

        def make_cache():
            return {
                "kind": "int4_kivi_dsl_paged",
                "jac_launch_order": ns["JAC_LAUNCH_ORDER"],
                "K_CODE_PAGES": _FakeTensor(uint8),
                "K_SCALE_PAGES": _FakeTensor(fp16),
                "K_ZERO_PAGES": _FakeTensor(fp16),
                "V_CODE_PAGES": _FakeTensor(uint8),
                "V_SCALE_PAGES": _FakeTensor(fp16),
                "V_ZERO_PAGES": _FakeTensor(fp16),
                "K_RESIDUAL_SCRATCH": _FakeTensor(fp16),
                "V_RESIDUAL_SCRATCH": _FakeTensor(fp16),
            }

        return ns, _FakeTensor, int32, int64, fp16, fp32, make_cache, recorded

    def test_out_of_range_seq_lens_value_fails_closed_before_dispatch(self):
        ns, T, int32, int64, fp16, fp32, make_cache, recorded = self._exec_stub()
        launch = ns["launch_kivi_dsl_decode_batched"]
        cache = make_cache()
        with self.assertRaisesRegex(ValueError, r"SEQ_LENS\[1\] must be in \[1, 4096\]"):
            launch(
                q=T(fp16),
                seq_lens=T(int32, values=[1, 5000]),
                block_table=T(int32),
                cache=cache,
                out=T(fp32),
                num_queries=2,
                stream=None,
            )
        self.assertEqual(recorded, [], "op must not launch on out-of-range SEQ_LENS")

    def test_zero_seq_lens_value_fails_closed(self):
        ns, T, int32, int64, fp16, fp32, make_cache, recorded = self._exec_stub()
        launch = ns["launch_kivi_dsl_decode_batched"]
        with self.assertRaisesRegex(ValueError, r"SEQ_LENS\[0\] must be in \[1, 4096\]"):
            launch(
                q=T(fp16),
                seq_lens=T(int32, values=[0, 4096]),
                block_table=T(int32),
                cache=make_cache(),
                out=T(fp32),
                num_queries=2,
                stream=None,
            )
        self.assertEqual(recorded, [])

    def test_in_range_seq_lens_value_reaches_the_op(self):
        ns, T, int32, int64, fp16, fp32, make_cache, recorded = self._exec_stub()
        launch = ns["launch_kivi_dsl_decode_batched"]
        cache = make_cache()
        launch(
            q=T(fp16),
            seq_lens=T(int32, values=[1, 4096]),
            block_table=T(int32),
            cache=cache,
            out=T(fp32),
            num_queries=2,
            stream=None,
        )
        self.assertEqual(len(recorded), 1)
        args = recorded[0]
        # launch order: q, seq_lens, block_table, cache tensors..., num_queries, out
        self.assertEqual(args[0].dtype, fp16)
        self.assertEqual(args[1].dtype, int32)
        self.assertEqual(args[2].dtype, int32)
        self.assertEqual(args[-2], 2)
        self.assertEqual(args[-1].dtype, fp32)

    def test_mixed_devices_fail_closed_before_dispatch(self):
        ns, T, int32, int64, fp16, fp32, make_cache, recorded = self._exec_stub()
        launch = ns["launch_kivi_dsl_decode_batched"]
        with self.assertRaisesRegex(RuntimeError, "span devices"):
            launch(
                q=T(fp16, device="cuda:0"),
                seq_lens=T(int32, values=[1, 2], device="cuda:1"),
                block_table=T(int32),
                cache=make_cache(),
                out=T(fp32),
                num_queries=2,
                stream=None,
            )
        self.assertEqual(recorded, [])


class PreflightArchHonestyTest(unittest.TestCase):
    """detect_cuda_arch must not silently pick among heterogeneous GPUs."""

    def _smi(self, stdout):
        proc = mock.Mock(returncode=0, stdout=stdout)
        with mock.patch.object(preflight.shutil, "which", return_value="/fake/nvidia-smi"), \
             mock.patch.object(preflight.subprocess, "run", return_value=proc):
            return preflight.detect_cuda_arch()

    def test_homogeneous_report_is_detected(self):
        self.assertEqual(self._smi("8.9\n8.9\n"), "sm_89")
        self.assertEqual(self._smi("8.0\n"), "sm_80")

    def test_heterogeneous_report_refuses_to_choose(self):
        self.assertIsNone(self._smi("8.0\n8.9\n"))

    def test_preflight_arch_detail_is_honest_about_build_target(self):
        from test_layout_and_scaffold import _write_minimal_vllm_tree
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_minimal_vllm_tree(root)
            with mock.patch.object(preflight, "git_head", return_value=None), \
                 mock.patch.object(preflight, "detect_cuda_arch", return_value="sm_89"):
                report = preflight.preflight(root)
        arch = [c for c in report["checks"] if c["name"] == "cuda_arch"][0]
        self.assertEqual(arch["status"], "pass")
        self.assertIn("build target", arch["detail"])
        self.assertIn("does not prove the runtime GPU", arch["detail"])
        self.assertIn(
            "torch-op launch boundary",
            report["integrity_boundary"],
        )


if __name__ == "__main__":
    unittest.main()
