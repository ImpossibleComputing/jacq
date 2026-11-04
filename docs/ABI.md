# ABI: versioning and compatibility

This document defines the compatibility contract between the three
independently built artifacts of the vLLM integration: DSL-generated CUDA
translation units, the scaffolded vLLM source tree, and the Python/torch
call sites that bind tensors.

## Version

`integrations/vllm_kivi/abi.py` defines:

```python
JACQ_KIVI_ABI_VERSION = 1   # bump on any tensor-shape, launch-order, or
                            # constant change; consumers must compare before
                            # binding tensors
```

`kernel_interface(seq_len)` returns the versioned, JSON-serialisable surface:
the required torch ops, the cache-writer selector values, and for each kernel
the generating spec plus the exact `jac_launch` tensor order.

## Hard boundaries (all fail closed)

Every check raises `UnsupportedAbiError` with the violated boundary; nothing
silently degrades or falls back.

| Boundary | Value | Check |
|---|---|---|
| vLLM version | `0.8.5` line, source head `3015d5634e74d59704e2b39bab0dbe2e6f86a38a` | `check_vllm_version()` — accepts exactly `0.8.5` plus non-numeric PEP 440 suffixes; lookalikes such as `0.8.50` fail closed. A version string does not prove the checked-out commit: resolve `git -C <vllm-root> rev-parse HEAD` and pass it as `source_head`. |
| CUDA arch | `sm_80` (A100), `sm_89` (RTX 4090) | `check_cuda_arch()` |
| Attention geometry | 40 query heads / 8 KV heads / head_dim 128 (Qwen2.5-Coder shape) | `check_geometry()` |
| Sequence length | ≤ 4096 tokens (124 old pages × 32 + 128 residual) | `kernel_interface()` |
| KV cache dtype | `int4_kivi` (group 32, nbits 4) | `KiviPageLayout.validate_for_vllm_lane()` |
| Block size | 32 (KIVI key groups must not straddle vLLM pages) | same |
| Cache-writer selector | `hand_cuda` or `jac_dsl` via `JAC_KIVI_CACHE_WRITE_KERNEL` | scaffolded backend |

The sequence-length cap is an ABI limit, not a performance recommendation.
Long-context use requires an explicit ABI lift: extend
`OLD_BLOCKS`/`NUM_BLOCKS`/`MAX_BLOCKS` in the kernel spec, regenerate the
translation unit, bump `JACQ_KIVI_ABI_VERSION`, and re-validate both arms of
any comparison.

## Tensor launch order (decode, ABI v1)

`Q, BLOCK_TABLE, K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, V_CODE_PAGES,
V_SCALE_PAGES, V_ZERO_PAGES, K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, SEQ_LEN,
O, cudaStream_t stream`

Shapes/dtypes per tensor are defined by
`integrations/vllm_kivi/bindings.py::KiviDecodeBindingAbi` (the single source
of truth). Residual storage is scratch-resident: there is no separate
`K_RESIDUAL`/`V_RESIDUAL` tensor in this ABI.

## Where the checks actually run

Fail-closed is a property of specific enforcement points, not of the helper
functions alone:

- **Preflight (build/apply time, automatic):** `source_patch.py --preflight`
  (and `--apply`, which runs it first) enforces the pinned git HEAD, scaffold
  anchors, and CUDA architecture against the actual checkout. `--skip-preflight`
  exists only for CPU tests against synthetic trees.
- **Runtime (serve time, inside the scaffolded backend):** model geometry,
  per-request shapes, `seq_len ≤ 4096`, block size 32, and the cache-writer
  selector are validated by the scaffolded vLLM attention backend when
  tensors are bound.
- **CPU tests (`tests/`):** cover the helper/preflight logic and the
  binding-vs-emitted-signature pin. They do not execute vLLM or CUDA and do
  not, by themselves, establish runtime compatibility.

## Integrity boundary

`abi.py` describes a callable interface. It does not allocate tensors, launch
CUDA, or validate numerical behaviour. A correct ABI match is a necessary
precondition for calling the kernels, not evidence of correctness; kernel
correctness requires separate GPU bitchecks and parity gates against a
reference implementation. A four-case synthetic direct-launch smoke passed
on an RTX 4090 on November 3, 2026; served validation is described in
[GPU validation](GPU_VALIDATION.md).
