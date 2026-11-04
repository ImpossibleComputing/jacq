# Current scope

Jacq's `.jac` model and kernel examples can be parsed and emitted on a CPU-only
machine. The repository tests cover compiler output, generated fixtures, and
the vLLM scaffold's ABI and preflight logic. A direct-launch RTX 4090 smoke
also passed four synthetic cases for the emitted paged KIVI kernel on November
3, 2026. A [vLLM serving run](VERIFIED_SERVING.md) also matched stock fp16
outputs across seven completion requests in each of two int4-KIVI writer
arms on one RTX 4090 and a 14B model.
A [public-source A100 follow-up](VERIFIED_SERVING.md#a100-follow-up-november-4-2026)
matched the same three cache configurations on three synthetic completions
and one chat request per arm under a 512-token serving configuration.

## Supported vLLM lane

- The integration patches the pinned vLLM 0.8.5 source head documented in
  [vLLM setup](VLLM_SETUP.md). It is an opt-in source build.
- ABI v1 accepts sequence lengths up to 4096 and block size 32.
- The vLLM KIVI path checks Qwen2.5-Coder attention geometry and CUDA compute
  capabilities `sm_80` (A100) and `sm_89` (RTX 4090). The launch boundary
  checks the actual tensor device.
- Batched prefill and sparse-retained decode are opt-in paths. See
  [ABI](ABI.md) for their tensor layouts and checks.

The serving result establishes output agreement for the prompts and settings
in [the evidence record](VERIFIED_SERVING.md). For further models, request
patterns, and performance comparisons, use the paired procedure in
[GPU validation](GPU_VALIDATION.md).
