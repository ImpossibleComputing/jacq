# Model-family checks on the AI Workstation

These checks used Jacq's standalone CUDA path for three model configurations and the pinned vLLM 0.8.5 source for serving. The short examples show how to run Jacq on a model; [summary.json](summary.json) contains the exact model revisions, synthetic prompts, token IDs, versions, and observed results. The table records the original `e44d7903` candidate. [The November 4 follow-up](../model-families-2026-11-04/README.md) adds early startup rejection to the unsupported vLLM cases without changing their Jacq KIVI support status.

| Model | Jacq standalone CUDA | Stock vLLM | Jacq int4-KIVI serving |
| --- | --- | --- | --- |
| TinyLlama 1.1B (Llama family) | 3 prompts × 20 raw-greedy tokens matched Hugging Face; longest prompt 400 tokens | One completion passed | Unsupported: head size 64 is rejected at startup |
| Qwen2.5 0.5B | 3 prompts × 20 raw-greedy tokens matched Hugging Face; longest prompt 399 tokens | One completion passed | Unsupported: head size 64 is rejected at startup |
| Qwen3 0.6B | No standalone path: the model needs Q/K normalization not expressed by these specs | One completion passed | Unsupported: 16/8 head geometry is rejected on the first request |
| Mistral 7B v0.1 | 3 prompts × 20 tokens matched Hugging Face; longest prompt 400 tokens | One completion passed | Unsupported: Mistral's attention path raises `Unsupported data type of kv cache: int4_kivi` on the first request |

The previously tested Qwen2.5-Coder-14B-AWQ 40/8 model remains the positive Jacq int4-KIVI serving example; see [the vLLM validation bundle](../vllm-2026-11-03/README.md). A stock-vLLM pass in this table shows only that the model can serve without Jacq KIVI. On this original candidate, the Qwen3 and Mistral KIVI errors reached the first request and returned HTTP 500. The follow-up candidate rejects them during startup.

## Run a standalone example

Build from the repository root, then choose one spec:

```sh
cargo build -p jac-cli
./target/debug/jacq --validate specs/models/tinyllama.jac
./target/debug/jacq --run specs/models/tinyllama.jac --arch sm_89 --max-new 20

./target/debug/jacq --validate examples/qwen2_5_0_5b.jac
./target/debug/jacq --run examples/qwen2_5_0_5b.jac --arch sm_89 --max-new 20

./target/debug/jacq --validate examples/mistral_7b_short_context.jac
./target/debug/jacq --run examples/mistral_7b_short_context.jac --arch sm_89 --max-new 20
```

`--validate` reads the `.jac` file without downloading weights. `--run` generates CUDA and a Hugging Face weight exporter, compiles with `nvcc`, downloads the model if needed, and runs inference. The Mistral model has 14.48 GB of weights and the exporter writes another roughly 14 GB of local fp16 files. Check disk and GPU memory before that run. We tested only inputs shorter than Mistral v0.1's 4096-token sliding window; these results do not establish sliding-window behavior.

The exporter now overrides a model repository's generation defaults to give a fair **raw greedy** reference: one argmax token at each step, fixed length, no repetition penalty, and no EOS stop. This matters for Qwen2.5, whose repository default includes `repetition_penalty: 1.1`, and for TinyLlama, which otherwise stops at EOS on the needle prompt. The change affects the reference comparison, not the CUDA runtime or a model's usual chat settings.

For three synthetic prompts after a `--run` has produced `generated/<model>/weights/`, use the bundled comparator with a locally cached snapshot:

```sh
export MODEL_PATH=/path/to/model/snapshots/<revision-from-summary.json>
export CUDA_VISIBLE_DEVICES=1
python3 validation/model-families-2026-11-03/compare_standalone.py \
  --model-snapshot "$MODEL_PATH" \
  --generated-dir "$PWD/generated/tinyllama" \
  --max-new 20 --raw-greedy
```

Change `--generated-dir` to `generated/qwen2_5_0_5b` or `generated/mistral_7b_short_context` for the other examples. The comparator loads the Hugging Face model, records its token IDs, releases its GPU memory, then runs Jacq; this matters for the 7B model on a 24 GB GPU. A result's `equal` field reports exact token-ID equality. The three prompts include short text, a 169–202-token retrieval prompt, and a 399–400-token longer prompt. Exact tokens are the gate here; the 0.0218 maximum teacher-forced NLL difference observed in Qwen diagnosis is recorded for context, not used as a replacement pass threshold.

## Reproduce the vLLM availability checks

Use the pinned vLLM source and an installed build with its usual dependencies. This probe starts a loopback server, sends one deterministic completion, records the response or error, and stops that server:

```sh
export VLLM_ROOT=/path/to/vllm-at-3015d5634e74d59704e2b39bab0dbe2e6f86a38a
export MODEL_PATH=/path/to/model/snapshots/<revision-from-summary.json>
export PYTHON_EXECUTABLE=/path/to/vllm-venv/bin/python
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_EXECUTABLE" validation/model-families-2026-11-03/probe_model_family_vllm.py \
  --model-snapshot "$MODEL_PATH" --vllm-root "$VLLM_ROOT" \
  --evidence-dir ./model-family-probe --label stock-check --kv-cache-dtype auto
```

Mistral used `--gpu-memory-utilization 0.8`; the smaller models used the probe's 0.3 default. Substituting `--kv-cache-dtype int4_kivi` reproduces the unsupported-model checks in the table; it is not a path to enabling those models. All four tested models were public, ungated, Apache 2.0 snapshots. Tests ran sequentially on GPU 1 (RTX 4090, sm_89), with no cloud GPU. No throughput, concurrency, streaming, large-context, TP=2, or other-GPU conclusion follows from this matrix.
