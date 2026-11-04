# AI Workstation serving validation

This bundle contains synthetic prompts, normalized responses, a portable serving runner, a CUDA boundary probe, and SHA-256 hashes of raw evidence retained in the private verification workspace. No model weights, credentials, or raw server logs are included.

## Results and provenance

`provenance.json` names three distinct Jacq commits. The original `7f3711c` run passed 15 completions and chat per arm at a configured 4096-token limit, but its separate 512-token attempt failed in the C++ block-table guard. Intermediate `112e2c8` passed seven completions and chat per arm at 512, but its guard was too strict for the published 124-entry table at 4096. Corrected `b396593` uses the actual old-page count. Its `corrected-b396593-responses.json` and `gpu-boundary-probe.json` record the final rerun. The response files intentionally retain only synthetic prompts, text, finish reasons, token counts, chat text, and KIVI op counts. The op tracer adds overhead; times are omitted.

Each response file has a stock fp16-cache arm and two int4-KIVI arms, one using the hand-CUDA writer and one using the Jacq DSL writer. Compare the `requests` arrays and `chat_text` fields across arms. The two KIVI arms' `op_counts` establish that the KIVI decode and selected writer actually executed. The GPU probe tests exact table widths around residual age-out and at 4096. The separate `raw-evidence-sha256.json` ties these normalized files to retained raw JSON, logs, build records, and binary hashes; its keys are relative to the private raw evidence directories listed in `provenance.json`.

## Reproduce on a compatible workstation

Use this combined Jacq candidate checkout for reproduction. Its runtime source, code generator, and specs are byte-identical to GPU-tested Jacq commit `b3965939f6c0b9037de0d027cd5898a2a8fcd3ff`; that SHA remains the historical evidence identity. Use vLLM commit `3015d5634e74d59704e2b39bab0dbe2e6f86a38a` and the cached model revision in `provenance.json`. The tested runtime was Python 3.12.13, PyTorch 2.6.0+cu124, Transformers 4.51.1, CUDA toolkit 12.4.131, driver 595.71.05, and one RTX 4090 (sm_89). Start with a pristine checkout of the pinned vLLM commit and the ordinary dependencies in Jacq's setup guide. Generate **all eight** CUDA variants referenced by CMake:

Set paths for your own machine:

```sh
export VLLM_ROOT=/path/to/pinned-vllm
export JACQ_SOURCE_ROOT=/path/to/jacq-inference
export MODEL_PATH=/path/to/Qwen2.5-Coder-14B-Instruct-AWQ/snapshots/eb3172f06a6d6b3a15f08947b0668d782e4d2d2c
export PYTHON_EXECUTABLE=/path/to/vllm-venv/bin/python
export CUDA_VISIBLE_DEVICES=1
cd "$JACQ_SOURCE_ROOT"
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" --preflight --cuda-arch sm_89 --json
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --sparse-retained
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --sparse-retained --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --batched
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --batched --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --cache-write
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --json --cache-write-batched
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" --apply --cuda-arch sm_89 --json
cd "$VLLM_ROOT"
env -u VLLM_USE_PRECOMPILED -u VLLM_PRECOMPILED_WHEEL_LOCATION "$PYTHON_EXECUTABLE" setup.py build_ext --inplace
```

Check torch-op registration, then run the boundary probe on the built extension:

```sh
cd "$JACQ_SOURCE_ROOT/validation/vllm-2026-11-03"
PYTHONPATH="$VLLM_ROOT" "$PYTHON_EXECUTABLE" -c 'import torch, vllm._C; assert hasattr(torch.ops._C, "kivi_decode_dsl")'
PYTHONPATH="$VLLM_ROOT" "$PYTHON_EXECUTABLE" probe_old_pages_gpu.py
```

For the same op-call evidence, instrument a disposable vLLM source checkout once before serving:

```sh
"$PYTHON_EXECUTABLE" instrument_ops.py
```

Run the three arms sequentially. The runner binds its server to loopback, uses the model cache offline, stops each server, disables prefix caching with the vLLM 0.8.5 flag, and writes full raw JSON and logs to a local directory:

```sh
export JACQ_TEST_EVIDENCE_DIR="$PWD/reproduced-512"
export JACQ_TEST_MAX_MODEL_LEN=512
export JACQ_TEST_WAVES=4
"$PYTHON_EXECUTABLE" run_serving.py
```

`synthetic-fixtures.json` lists the exact requests. The runner sets seed 42 and temperature 0. Its fixed serving flags include V1, eager mode, block size 32, TP=1, max sequences 2, and `--no-enable-prefix-caching`. The 512-configured run uses 93- and 219-token prompts; it is not a 512-token-input test. The corrected guard's 4096-token capacity was exercised through the direct CUDA op probe, not a full 4096-token serving request. These results do not establish performance, concurrency, streaming, other model geometries, TP=2, or A100 behavior.
