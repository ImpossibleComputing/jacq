# vLLM opt-in setup (int4-KIVI serving lane)

This **opt-in source integration** builds Jacq's DSL-lowered paged int4-KIVI
kernels into a pinned vLLM checkout. Start with the
[`.jac` quick start](QUICK_START.md) if you are new to the compiler.

## Requirements

- A CUDA-capable machine (validated architectures: A100 `sm_80`, RTX 4090
  `sm_89` — others fail closed in `abi.py`).
- CUDA toolkit (`nvcc`), PyTorch built with CUDA, `torch`/`transformers`/
  `numpy` for weight handling.
- vLLM source checkout on the `0.8.5` line, source head
  `3015d5634e74d59704e2b39bab0dbe2e6f86a38a`. Other versions are rejected by
  the ABI checks (`check_vllm_version`), and the scaffold patch is anchored
  against this exact source.

## Steps

Run everything from this repository's root.

1. **Build the compiler and run the CPU checks first:**

   ```bash
   cargo build --workspace
   cargo test --workspace
   python3 -m unittest discover -s tests -v
   ```

2. **Prepare the pinned vLLM checkout.** All commands below run from this
   repository's root — do not `cd` into the vLLM tree; `VLLM_ROOT` carries
   the path instead, so the Jacq commands keep working:

   ```bash
   VLLM_ROOT=/path/to/vllm-checkout
   git clone https://github.com/vllm-project/vllm.git "$VLLM_ROOT"
   git -C "$VLLM_ROOT" checkout 3015d5634e74d59704e2b39bab0dbe2e6f86a38a
   git -C "$VLLM_ROOT" rev-parse HEAD   # must print the pinned SHA above
   ```

   The `rev-parse` line is a manual sanity check; step 4's `--preflight`
   enforces it automatically before applying (HEAD must equal the pinned
   SHA). The scaffold's anchor check alone verifies only that the vLLM
   *source files* match the pinned text — a drifted Git HEAD with
   coincidentally matching anchors would pass anchoring, which is why the
   preflight checks both. For the same reason, pass the resolved HEAD as
   `source_head` to `check_vllm_version()` if you use the ABI helpers
   programmatically; a version string alone does not prove which commit is
   checked out.

3. **Materialize the DSL-generated CUDA kernels into the vLLM tree.**
   The scaffolded CMake target includes eight generated translation units:
   base, TP=2-local, sparse-retained, and batched decode (each with the
   corresponding TP=2-local variant), plus single and batched cache writers.
   `--tp2-local` emits both a base and a rank-local artifact:

   ```bash
   python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --tp2-local
   python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --sparse-retained --tp2-local
   python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --batched --tp2-local
   python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --cache-write
   python3 -m integrations.vllm_kivi.dsl_decode --vllm-root "$VLLM_ROOT" --emit --cache-write-batched
   ```

   This CPU-only step regenerates each kernel from its `.jac` spec and copies
   the CUDA sources to `csrc/kivi/`. Add `--json` to inspect each artifact's
   source, destination, and serving environment fields.

4. **Run the fail-closed preflight, then apply the source scaffold patch.**

   ```bash
   # preflight: pinned git HEAD + scaffold anchors + CUDA arch; nonzero exit on failure
   python3 -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" \
       --preflight --cuda-arch sm_89          # or sm_80 for A100

   # anchor-checked, idempotent apply; runs the same preflight automatically
   # and refuses on failure (there is a --skip-preflight escape for CPU test
   # harnesses only — never use it on a real checkout)
   python3 -m integrations.vllm_kivi.source_patch --vllm-root "$VLLM_ROOT" --apply
   ```

   (Invoke as a module with `python3 -m`: the script imports the
   `integrations` package, so a direct `python3
   integrations/vllm_kivi/source_patch.py ...` from repo root fails with
   `ModuleNotFoundError` unless the repo root is on `PYTHONPATH`.)

   What the preflight does and does not enforce:

   - **Enforced automatically**: the checkout's git HEAD must equal the
     pinned source head (a source tarball fails — it cannot prove its
     commit), all scaffold anchors must match (`--json` prints the full
     anchor report), and the CUDA architecture must be one of the validated
     set (`--cuda-arch`, or auto-detected via `nvidia-smi` when it reports
     exactly one compute capability; heterogeneous multi-GPU hosts are not
     silently resolved to one arch — pass `--cuda-arch` explicitly).
   - **Build target, not runtime proof**: the architecture check validates
     a declared/detected build target only; it does not prove which GPU a
     request actually serves. The actual serving device is enforced
     fail-closed at the C++ torch-op launch boundary: every
     `kivi_*` op reads the compute capability of the device its tensors
     are bound to, accepts only sm_80/sm_89, and rejects mixed-device
     launches (see `check_launch_device` in
     `integrations/vllm_kivi/source_patch.py`'s `KIVI_BINDING_SOURCE`).
   - **Model compatibility**: startup accepts `int4_kivi` only for the
     Qwen2ForCausalLM full-attention configuration with 40 attention heads,
     8 KV heads, head size 128, fp16 activations, V1 FlashAttention, TP=1 or
     TP=2, PP=1, block size 32, maximum model length at most 4096, and prefix
     caching off. An unsupported model such as Mistral exits with a clear
     configuration error before serving. A stock vLLM request with
     `--kv-cache-dtype auto` does not establish Jacq KIVI support.
   - **Not verified by preflight**: model geometry, per-request shapes, and
     `seq_len ≤ 4096`. The startup check covers configured compatibility;
     the attention backend still checks bound tensors and request shapes.
     Passing preflight or startup alone does not prove kernel correctness or
     serving behavior.

   The scaffold makes vLLM accept `--kv-cache-dtype int4_kivi`, enforce
   `--block-size 32`, add the KIVI attention spec with explicit byte
   accounting, wire the per-layer int4 tensor bundle allocation, and add
   the `csrc/kivi/*.cu` sources to the extension build. Use `--diff` to
   preview the full unified diff (the frozen equivalent is checked in at
   `integrations/vllm_kivi/patches/vllm-0.8.5-int4-kivi-scaffold.patch`,
   regenerated from the pinned head).

5. **Build the patched vLLM from source** (GPU machine). CUDA architecture
   is a build-time compatibility concern, decided in this step — it is not a
   serving flag:

   ```bash
   cd "$VLLM_ROOT" && python3 setup.py develop
   # or the repo's documented build; requires nvcc. Build on a machine whose
   # GPU is one of the validated architectures (sm_80 / A100, sm_89 / RTX
   # 4090 — see docs/ABI.md; the preflight helper is
   # integrations.vllm_kivi/abi.py::check_cuda_arch). nvcc encodes the
   # architecture at compile time via -arch/-gencode flags per the vLLM build
   # system's defaults.
   cd -   # back to the Jacq repo root
   ```

6. **Serve with the lane enabled.** Opt-in flags/env, all of which fail
   closed on unsupported shapes (seq_len ≤ 4096, block size 32). Note there
   is no `--arch` argument on the serving command; the GPU architecture was
   fixed at build time in step 5, and vLLM has no runtime flag for it:

   ```bash
   export VLLM_USE_V1=1
   export JAC_KIVI_CACHE_WRITE_KERNEL=hand_cuda   # use jac_dsl for the verified DSL writer
   export JAC_KIVI_BATCHED_PREFILL=0               # default off
   export JAC_KIVI_SPARSE_RETAINED_DECODE=0        # default off
   python3 -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
       --kv-cache-dtype int4_kivi \
       --block-size 32 \
       --max-model-len 512 \
       --max-num-batched-tokens 512 \
       --max-num-seqs 2 \
       --tensor-parallel-size 1 \
       --gpu-memory-utilization 0.7 \
       --enforce-eager \
       --no-enable-prefix-caching \
       --disable-log-requests
   ```

   This is the model, context setting, and request configuration used in the
   [verified serving run](VERIFIED_SERVING.md). Record the actual build,
   device, and output when reproducing it; see
   [GPU validation](GPU_VALIDATION.md).

7. **Compare serving results.** First compare the hand-CUDA and Jacq DSL
   int4-KIVI writers, then compare stock fp16 and int4-KIVI cache on the same
   binary. [GPU validation](GPU_VALIDATION.md) gives the paired commands and
   measurements to record.

## Rollback

The scaffold patch is a unified diff of specific vLLM files; revert with
`git checkout -- <files> && git clean -fd csrc/kivi` inside the vLLM checkout,
or simply discard the checkout. The standalone compiler path is unaffected by
any of this.
