# Troubleshooting

## Compiler / codegen

**Compiler binary** — run `cargo build --workspace` from the repository root;
the executable is `target/debug/jacq` (`cargo run --bin jacq -- --help`).

**Golden/snapshot test failures after changing a kernel spec or codegen** —
the snapshot fixtures in `crates/jac-codegen-cuda/tests/snapshots/` pin exact
output. Regenerate expectations deliberately, review the diff, and keep the
change consistent with a bump to `JACQ_KIVI_ABI_VERSION` in
`integrations/vllm_kivi/abi.py` if it affects the serving lane.

**`--emit-model` produces files but `--run` fails at nvcc** — the standalone
runtime path needs `nvcc` on PATH and an NVIDIA GPU; check `jacq --doctor`.
`--arch` must match the GPU (`sm_80` A100, `sm_89` RTX 4090).

**Weight export fails (`ModuleNotFoundError: torch`)** — the generated
`<name>_export.py` needs `torch`, `transformers`, `numpy` (HuggingFace model
download). These are only needed for `--run`, not for build/tests.

## vLLM integration

**`source_patch.py --apply` errors on anchor mismatch** — the checkout is not
the pinned vLLM source head (`3015d5634…`). The patch intentionally fails
closed rather than patching approximately. Re-checkout the pinned head or
re-anchor the patch deliberately (which requires re-validating the ABI).

**`UnsupportedAbiError: seq_len ... exceeds the KIVI ABI cap`** — request
contexts are above 4096 tokens. This is a hard ABI v1 limit; lower
`--max-model-len` or lift the ABI explicitly (see [ABI.md](ABI.md)).

**`UnsupportedAbiError` on vLLM version / CUDA arch / geometry** — the
failure message names the violated boundary. The supported set is exactly the
vLLM `0.8.5` line (lookalikes such as `0.8.50` fail closed), the pinned
source head, sm_80/sm_89, and the Qwen2.5-Coder attention shape; anything
else fails closed by design. Version strings alone do not prove the checkout:
run `git -C "$VLLM_ROOT" rev-parse HEAD` and compare it to the pinned head
before building — the scaffold's anchor check validates source text, not the
Git HEAD.

**C++ op error `... reports compute capability ... outside the validated set
sm_80/sm_89` or `tensors must all live on one CUDA device`** — the scaffolded
torch ops enforce the ACTUAL runtime device: every `kivi_*` launch reads the
compute capability of the device its tensors are bound to (accepting only
sm_80/sm_89) and rejects mixed-device bundles. The preflight's CUDA-arch check
is a declared/detected build target only; this runtime guard is what actually
fails closed when a request lands on an unvalidated GPU (including a second
GPU of a different architecture on a heterogeneous multi-GPU host).

**`ModuleNotFoundError: No module named 'integrations'` when running
`source_patch.py`** — the script imports the `integrations` package, so run
it as a module from the Jacq repo root:
`python3 -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT"
--apply` (or put the repo root on `PYTHONPATH`). The `dsl_decode.py` script
is stdlib-only and runs directly either way.

**`torch.ops._C.kivi_decode_dsl` or another KIVI op is missing** — the
scaffolded extension did not build or a generated `csrc/kivi/*.cu` source is
missing. Run all generation commands in [vLLM setup](VLLM_SETUP.md), rebuild
vLLM from source, and check the registration with
`python3 -c "import vllm._C; import torch; print(torch.ops._C.kivi_decode_dsl)"`.

**JAC_KIVI_CACHE_WRITE_KERNEL set to something else** — only `hand_cuda`
(default) and `jac_dsl` are accepted; anything else fails closed.

**Unexpected served results** — compare the same model, hardware, build,
prompts, and settings across both arms and keep the raw responses. The
[GPU validation guide](GPU_VALIDATION.md) lists the checks to repeat.
