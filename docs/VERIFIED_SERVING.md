# Verified vLLM serving runs

On November 3, 2026, the patched vLLM 0.8.5 build served
`Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` with three cache configurations on one
RTX 4090: stock fp16, int4-KIVI with the hand-CUDA cache writer, and
int4-KIVI with the Jacq DSL cache writer. The corrected Jacq fix under test was
`b3965939f6c0b9037de0d027cd5898a2a8fcd3ff`, based on release candidate
`7f3711cc860dd78f9abf6ab0d2252a47e4ce209d`.

The [validation bundle](../validation/vllm-2026-11-03/README.md) contains
portable commands, synthetic requests, normalized responses, the CUDA
boundary probe, and checksums for the retained raw logs. It keeps the
original, intermediate, and corrected runs distinct.

## Result

Each arm completed seven deterministic completion requests and one chat
request with `--max-model-len 512`. Both int4-KIVI arms matched stock fp16 on
the completion text, finish reason, and token counts for all seven requests;
the chat response was `READY` in every arm. The 219-token retrieval prompt
returned ` The secret code is LIME42` in all three arms.

| Request | Prompt tokens | Completion text in all three arms |
| --- | ---: | --- |
| short one-token | 93 | ` blue` |
| long one-token | 219 | ` The` |
| retrieval | 219 | ` The secret code is LIME42` |
| alternating 1 | 93 | ` blue.` |
| alternating 2 | 219 | ` The secret` |
| alternating 3 | 93 | ` blue.` |
| alternating 4 | 219 | ` The secret` |

An instrumented extension recorded 57,744 `kivi_decode_dsl` calls in each
int4-KIVI arm. It also recorded 57,744 calls to each arm's selected writer:
`kivi_cache_write_dsl` for hand-CUDA and `kivi_cache_write_jac_dsl` for Jacq.
This confirms that the requests used the intended kernels.

The rebuilt CUDA op also exercised block-table capacity at the 128-token
residual boundary and at the 4096-token ABI ceiling. It accepted widths 0
at sequence lengths 128, 129, and 159; width 1 at 160 and 191; width 2 at
192; and width 124 at 4096, returning finite output. It rejected widths 0
at 160, 1 at 192, and 123 at 4096 before launch. The exact cases are in
[`gpu-boundary-probe.json`](../validation/vllm-2026-11-03/gpu-boundary-probe.json).

## Tested configuration

- vLLM source commit `3015d5634e74d59704e2b39bab0dbe2e6f86a38a`,
  patched and rebuilt with the corrected Jacq C++ binding; runtime version
  `0.8.5.post2.dev0+g3015d5634.d20261103`.
- Model snapshot `eb3172f06a6d6b3a15f08947b0668d782e4d2d2c`;
  40 attention heads, 8 KV heads, AWQ weights.
- One RTX 4090 (`sm_89`), tensor parallel size 1, PyTorch `2.6.0+cu124`,
  CUDA toolkit 12.4.131, Python 3.12.13, Transformers 4.51.1, and vLLM V1
  enabled. Driver version was 595.71.05; the rebuilt `_C` extension's SHA-256
  was `a39cdedc80b6d6473c8caacbcc722504528d6e8d26373e346e5f8c33f8ca4d68`.
- All arms used block size 32, maximum model length 512, maximum batched
  tokens 512, maximum sequences 2, eager execution, and prefix caching off.
  The two KIVI arms differed only in `JAC_KIVI_CACHE_WRITE_KERNEL`.

The fixed build corrected the block-table check: a 512-token server supplies
16 table entries, while the original check demanded 124 for every request.
The corrected guard requires only full 32-token pages aged past the
128-token residual: `floor(max(seq_len - 128, 0) / 32)`. It accepts a
124-entry table at 4096 while rejecting one that is too short for the old
pages a request will read.

This run covers sequential requests with 93- and 219-token prompts under a
512-token configuration. The 4096 boundary was checked by a direct CUDA op
probe, not a full-model 4096-token request. Streaming, concurrent serving,
other models or GPUs, and performance were outside this run.

## A100 follow-up (November 4, 2026)

The [A100 evidence bundle](../validation/a100-2026-11-04/README.md) records a
second GPU architecture using public Jacq commit
`1c797da0ec1c67c6b987a984a5f4c304cc52c77a`, pinned vLLM source commit
`3015d5634e74d59704e2b39bab0dbe2e6f86a38a`, and model snapshot
`eb3172f06a6d6b3a15f08947b0668d782e4d2d2c` of
`Qwen/Qwen2.5-Coder-14B-Instruct-AWQ`. It ran on one A100 80 GB (`sm_80`)
with CUDA toolkit 12.4.131, PyTorch `2.6.0+cu124`, and Python 3.11.10.

With a 512-token configured context, three synthetic completion requests
(93, 219, and 219 prompt tokens) and one chat request per arm returned
identical text, finish reasons, and token counts for stock fp16 caching,
int4-KIVI with the hand-CUDA writer, and int4-KIVI with the Jacq DSL writer.
The captured operation traces record 27,600 Jacq decode calls in each KIVI
arm and 27,600 calls to the selected writer in each arm. The DSL writer arm
therefore exercised both generated decode and generated cache-write ops.

A separate direct CUDA op probe accepted the required block-table widths at
sequence lengths 128, 129, 159, 160, 191, 192, and 4096, and rejected three
insufficient widths. The 4096 case was not a full-model serving request.
This A100 run does not measure performance or exercise TP=2, concurrency,
streaming, other models, or longer-context serving.
