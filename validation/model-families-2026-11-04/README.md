# Jacq KIVI model compatibility at startup

This follow-up to the [model-family matrix](../model-families-2026-11-03/README.md) changes unsupported-model handling in the pinned vLLM 0.8.5 scaffold. It checks `int4_kivi` compatibility while creating `VllmConfig`, before model weights load or the server accepts requests. The original model tests and their first-request HTTP 500 logs remain attributed to candidate `e44d7903ea1b2cd7052ba09cd9c1b937d223e130`. [summary.json](summary.json) records this follow-up's exact responses, versions, source hashes, and checksums of the retained raw logs.

| Model and mode | Result on the follow-up build |
| --- | --- |
| Mistral 7B v0.1, Jacq `int4_kivi` | Server exited 1 before readiness with `ValueError: int4_kivi currently supports only Qwen2ForCausalLM ... got architectures=['MistralForCausalLM'], model_type='mistral'`. No HTTP 500 request. |
| Qwen3 0.6B, Jacq `int4_kivi` | Same startup rejection naming `Qwen3ForCausalLM` and `qwen3`. No HTTP 500 request. |
| Qwen2.5-Coder-14B-Instruct-AWQ, stock fp16 vs Jacq `int4_kivi` with DSL writer | Three completion requests matched exactly on text, finish reason, and token counts; both chat replies were `READY`. The KIVI arm recorded 27,600 `kivi_decode_dsl` and 27,600 `kivi_cache_write_jac_dsl` calls. |

The Mistral standalone CUDA and stock vLLM results from the original matrix remain valid, as do the TinyLlama and small Qwen2 standalone results. A standalone or stock vLLM pass does not imply Jacq KIVI support. The positive KIVI check covers only the listed Qwen14B model on one RTX 4090, TP=1, 512-token server configuration, and short sequential requests. The startup guard accepts only Qwen2ForCausalLM full attention with global 40/8 heads, head size 128, fp16, V1 FlashAttention, TP=1 or 2, PP=1, block size 32, maximum model length 1–4096, and prefix caching off. Acceptance of a configuration is not a serving result for untested TP=2, models, GPUs, or context lengths.

## Reproduce on an isolated build

Use a pristine checkout of vLLM commit `3015d5634e74d59704e2b39bab0dbe2e6f86a38a`. Follow [vLLM setup](../../docs/VLLM_SETUP.md) to emit all required CUDA sources and build the checked-in scaffold. Use locally cached model snapshots at the revisions in [summary.json](summary.json). The following commands use the bundled probes and stop only the servers they start:

```sh
export VLLM_ROOT=/path/to/pinned-patched-vllm
export PYTHON_EXECUTABLE=/path/to/vllm-venv/bin/python
export CUDA_VISIBLE_DEVICES=1
export MISTRAL_PATH=/path/to/Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
export QWEN3_PATH=/path/to/Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
export MODEL_PATH=/path/to/Qwen2.5-Coder-14B-Instruct-AWQ/snapshots/eb3172f06a6d6b3a15f08947b0668d782e4d2d2c

"$PYTHON_EXECUTABLE" validation/model-families-2026-11-03/probe_model_family_vllm.py \
  --model-snapshot "$MISTRAL_PATH" --vllm-root "$VLLM_ROOT" \
  --evidence-dir ./startup-probe --label mistral-kivi --kv-cache-dtype int4_kivi \
  --gpu-memory-utilization 0.8
"$PYTHON_EXECUTABLE" validation/model-families-2026-11-03/probe_model_family_vllm.py \
  --model-snapshot "$QWEN3_PATH" --vllm-root "$VLLM_ROOT" \
  --evidence-dir ./startup-probe --label qwen3-kivi --kv-cache-dtype int4_kivi

VLLM_ROOT="$VLLM_ROOT" MODEL_PATH="$MODEL_PATH" \
PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" JACQ_TEST_EVIDENCE_DIR=./qwen14b-probe \
JACQ_TEST_MAX_MODEL_LEN=512 JACQ_TEST_WAVES=0 \
JACQ_TEST_ARMS=baseline_fp16,kivi_jac_dsl \
python3 validation/vllm-2026-11-03/run_serving.py
```

The negative probe reports `status: failed` because the server intentionally exits; inspect its `.server.log` for the named compatibility `ValueError`. The positive runner exits zero only if both arms serve. Compare its JSON responses and check that `kivi_jac_dsl.json` has positive `kivi_decode_dsl` and `cache_write:kivi_cache_write_jac_dsl` op counts. The raw workstation logs are retained locally outside the repository under `early-guard-evidence/`; their SHA-256 values are in [summary.json](summary.json).
