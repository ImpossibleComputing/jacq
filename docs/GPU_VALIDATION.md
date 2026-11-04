# GPU validation

The emitted paged KIVI kernel passed a direct-launch smoke on an RTX 4090
with four synthetic cases on November 3, 2026. This checks the generated
CUDA launch path. A [512-configured vLLM serving run](VERIFIED_SERVING.md)
has also passed for one 14B model on an RTX 4090. A
[smaller A100 follow-up](VERIFIED_SERVING.md#a100-follow-up-november-4-2026)
used the public Jacq release with the same model and three synthetic
completions plus one chat request per arm. The steps below show how to repeat
and extend these checks.

Use one idle NVIDIA GPU for each paired measurement. Record the Jacq commit,
vLLM commit/build, GPU, model revision, commands, settings, and raw outputs.

## 1. Build the pinned vLLM extension

From the Jacq repository root, set `VLLM_ROOT` to an absolute checkout path.
The source patch targets vLLM commit
`3015d5634e74d59704e2b39bab0dbe2e6f86a38a`:

```bash
VLLM_ROOT=/path/to/vllm-checkout
git clone https://github.com/vllm-project/vllm.git "$VLLM_ROOT"
git -C "$VLLM_ROOT" checkout 3015d5634e74d59704e2b39bab0dbe2e6f86a38a
python3 -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" \
  --preflight --cuda-arch sm_89  # use sm_80 on A100

# CMake requires all eight generated CUDA files; --tp2-local emits both variants.
python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --tp2-local
python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --sparse-retained --tp2-local
python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --batched --tp2-local
python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --cache-write
python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --cache-write-batched

python3 -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" --apply
cd "$VLLM_ROOT"
python3 -m pip install -e .
python3 -c "import vllm._C; import torch; print(torch.ops._C.kivi_decode_dsl)"
```

The op is registered under `torch.ops._C`. See
[vLLM setup](VLLM_SETUP.md) for the full requirements and ABI behavior.

## 2. Compare the two int4-KIVI cache writers

This gate compares the **hand-CUDA writer with the Jacq DSL writer** in the
same patched int4-KIVI vLLM build. It does not compare stock fp16 caching.
Start one server at a time with identical flags and the same model/GPU:

```bash
export JAC_KIVI_BATCHED_PREFILL=0
export JAC_KIVI_SPARSE_RETAINED_DECODE=0
export JAC_KIVI_CACHE_WRITE_KERNEL=hand_cuda  # first arm; use jac_dsl for second
export VLLM_USE_V1=1
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
  --kv-cache-dtype int4_kivi --block-size 32 --max-model-len 512 \
  --max-num-batched-tokens 512 --max-num-seqs 2 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 --enforce-eager \
  --no-enable-prefix-caching --disable-log-requests
```

Send identical deterministic prompts with `max_tokens=1`, including one
prompt below and one above the 160-token KIVI age-out boundary. Record exact
output text, finish reason, and token counts for each arm. Exact one-token
output agreement is the initial served parity check; include the raw HTTP
responses and server logs in the result artifact.

## 3. Compare stock and Jacq serving

On the same vLLM binary, GPU, model, prompts, and request settings, compare
stock fp16 KV cache with the int4-KIVI lane. Keep `--block-size 32` in both
arms and change only `--kv-cache-dtype` (`auto`/fp16 versus `int4_kivi`).
Record exact outputs and a quality check, latency,
completed requests per second, and peak GPU memory. If testing capacity,
record both arms' limiting failure kind before reporting a ratio. Keep the
raw requests, responses, measurements, and environment with the result.

The one-token parity check and the stock-cache comparison answer different
questions; report their results separately.
