# Jacq on A100: public-source verification

This bundle contains synthetic request fixtures, normalized responses, CUDA boundary-probe results, source and model hashes, and a reproducible command sequence. It contains no model weights, pod address, account data, or raw server logs. The `raw-evidence-sha256.json` file ties the normalized result to the full logs retained by the test operator.

## Observed result

On one NVIDIA A100 SXM4 80 GB (`sm_80`), the stock fp16-cache arm, the int4-KIVI hand-CUDA writer arm, and the int4-KIVI Jacq DSL writer arm returned identical text, finish reasons, and token counts for three completions and one chat request. The outputs were ` blue`, ` The`, ` The secret code is LIME42`, and chat `READY`. Each KIVI arm executed 27,600 `kivi_decode_dsl` calls. The Jacq arm executed 27,600 `kivi_cache_write_jac_dsl` calls; the hand-CUDA arm executed 27,600 `kivi_cache_write_dsl` calls. See `normalized-responses.json` and run `python3 verify.py` for the exact assertions.

The configured context limit was 512 tokens; the tested prompt lengths were 93 and 219 tokens. The direct CUDA probe accepted the required old-page table widths at lengths 128, 129, 159, 160, 191, 192, and 4096, and rejected three insufficient widths. The 4096 case is a direct op probe, not a 4096-token serving request. Tracing adds overhead, so these results make no performance claim. They do not establish support for other models, TP=2, long-context serving, concurrency, or streaming.

## Evidence identity and the 32B diagnostic

The retained raw `generated-sha256.txt` has SHA-256 `348317f156086030168972988480b9138b874164dd726175d7dc60b66ff9a102`. This bundle's `kivi-cuda-source-sha256.txt` has SHA-256 `6804b3905752954efff2292cfeb75a2dba14c6622e81860ce58770132cfbe236`. They are intentionally **different text files**: the portable copy removes the pod's absolute source-root prefix from each path, leaving `csrc/kivi/...`. All nine source-file digests and basenames match exactly. Eight sources were emitted from `.jac`; the ninth, `cache_write_kivi_dsl.cu`, is the hand-CUDA writer included by the scaffold despite its historical filename. `source-hash-normalization.json` records the transformation; `python3 verify.py --raw-dir /path/to/retained/evidence` checks the mapping against the raw file when available.

The direct GPU boundary probe constructs **synthetic** Q/K/V tensors with 40 attention heads, 8 KV heads and head size 128. Its error text says “Qwen2.5-Coder-32B” because that literal is in the public `source_patch.py` CUDA binding's shape and block-table checks. It is an older geometry label, not a model identification. The serving arms instead loaded the pinned [Qwen2.5-Coder-14B-Instruct-AWQ revision](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ/tree/eb3172f06a6d6b3a15f08947b0668d782e4d2d2c): the retained server logs show the local model directory, V1 AWQ-Marlin configuration, fp16 activations, and three safetensors shards loaded. `model-snapshot-crosscheck.json` compares the retained config, tokenizer, and all three shard hashes with that public revision. Its config has 48 layers and 40/8 heads; sharing the kernel's 40/8 geometry does not make it a 32B model.

## Reproduce

Use one A100 (`sm_80`) with CUDA toolkit 12.4 and enough disk for a source build and the public 9.4 GB model. The test used official RunPod image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04@sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`, Python 3.11.10, driver 595.91.07, and Rust 1.98.1. Create an isolated Python 3.11 virtual environment first. Install Rust 1.98.1 with `cargo` on `PATH`; the `.jac` emission commands invoke it. Set paths for your machine:

```sh
export JACQ_ROOT=/path/to/jacq
export VLLM_ROOT=/path/to/vllm
export MODEL_PATH=/path/to/public-qwen-awq-model
export PYTHON_EXECUTABLE=/path/to/venv/bin/python
export HF_HOME=/path/to/huggingface-cache
export PATH="$(dirname "$PYTHON_EXECUTABLE"):/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 CUDA_VISIBLE_DEVICES=0

git clone https://github.com/ImpossibleComputing/jacq.git "$JACQ_ROOT"
git -C "$JACQ_ROOT" checkout 1c797da0ec1c67c6b987a984a5f4c304cc52c77a
git clone https://github.com/vllm-project/vllm.git "$VLLM_ROOT"
git -C "$VLLM_ROOT" checkout 3015d5634e74d59704e2b39bab0dbe2e6f86a38a

"$PYTHON_EXECUTABLE" -m pip install torch==2.6.0 torchvision==0.21.0 \
  torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_EXECUTABLE" -m pip install -r "$VLLM_ROOT/requirements/build.txt" \
  -r "$VLLM_ROOT/requirements/cuda.txt"
"$PYTHON_EXECUTABLE" -m pip install transformers==4.51.1 setuptools==80.9.0

cd "$JACQ_ROOT"
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.source_patch \
  --vllm-root "$VLLM_ROOT" --preflight --cuda-arch sm_80 --json
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --sparse-retained
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --sparse-retained --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --batched
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --batched --tp2-local
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --cache-write
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.dsl_decode \
  --vllm-root "$VLLM_ROOT" --emit --json --cache-write-batched
"$PYTHON_EXECUTABLE" -m integrations.vllm_kivi.source_patch \
  --vllm-root "$VLLM_ROOT" --apply --cuda-arch sm_80 --json

cd "$VLLM_ROOT"
env -u VLLM_USE_PRECOMPILED -u VLLM_PRECOMPILED_WHEEL_LOCATION \
  "$PYTHON_EXECUTABLE" setup.py build_ext --inplace
"$PYTHON_EXECUTABLE" setup.py egg_info
PYTHONPATH="$VLLM_ROOT" "$PYTHON_EXECUTABLE" -c \
  'import torch, vllm._C; assert hasattr(torch.ops._C, "kivi_decode_dsl"); assert hasattr(torch.ops._C, "kivi_cache_write_jac_dsl")'

HF_HOME="$HF_HOME" MODEL_PATH="$MODEL_PATH" "$PYTHON_EXECUTABLE" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    revision="eb3172f06a6d6b3a15f08947b0668d782e4d2d2c",
    local_dir=os.environ["MODEL_PATH"],
)
PY

cd "$JACQ_ROOT/validation/vllm-2026-11-03"
PYTHONPATH="$VLLM_ROOT" "$PYTHON_EXECUTABLE" probe_old_pages_gpu.py
VLLM_ROOT="$VLLM_ROOT" "$PYTHON_EXECUTABLE" instrument_ops.py
export PYTHONPATH="$VLLM_ROOT"
export JACQ_TEST_EVIDENCE_DIR=/path/to/test-output
export JACQ_TEST_MAX_MODEL_LEN=512 JACQ_TEST_WAVES=0
export JACQ_TEST_ARMS=baseline_fp16,kivi_jac_dsl
"$PYTHON_EXECUTABLE" run_serving.py
export JACQ_TEST_ARMS=kivi_hand_cuda
"$PYTHON_EXECUTABLE" run_serving.py
```

For an Ada GPU (`sm_89`), change `--cuda-arch sm_80` to `sm_89` and `TORCH_CUDA_ARCH_LIST=8.0` to `8.9`. Check the actual GPU capability, driver, and free VRAM before building. This bundle reports only the A100 run; an Ada result requires its own serving and op evidence.
