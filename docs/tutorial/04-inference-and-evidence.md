# 4. Run and evaluate inference

**About 20 minutes to read; GPU setup time varies.** Lessons 1–3 generated
source on a CPU. This lesson explains what happens when source and model
weights meet a GPU, then offers an optional run. Use the [quick start](../QUICK_START.md)
for the standalone prerequisites and the [vLLM setup guide](../VLLM_SETUP.md)
for the separate serving integration.

## From a prompt to one more token

1. A tokenizer turns prompt text into token IDs. A token is a piece of text,
   not necessarily a word.
2. *Prefill* processes the prompt tokens. Each layer builds keys and values
   used by attention.
3. *Decode* predicts the next token, appends it to the sequence, and repeats.
   Greedy sampling chooses the highest-scoring candidate at each step.

The learned numeric arrays are *weights*. `examples/tinyllama.jac` names a
weight source; `--emit-model` writes an exporter and a CUDA runtime; the
optional `--run` step compiles that runtime, fetches/exports weights, then
executes it. The exporter may use the normal Hugging Face download cache;
exported weights go under the generated model directory.

On a machine with `nvcc`, a supported NVIDIA GPU, Python packages listed in
the [quick start](../QUICK_START.md), and access to the named model, return
to the repository root and run. The `--arch` value is the CUDA instruction
target for the GPU: `sm_89` for an RTX 4090, or `sm_80` for an A100.

```sh
cd "$JACQ_REPO"
./target/debug/jacq --doctor
./target/debug/jacq --run examples/tinyllama.jac --arch sm_89 --max-new 20
```

`--run` prints a reference continuation and generated token IDs for
comparison. Generated sources and exported weights are written under
`generated/tinyllama/`. This optional step downloads model
data and runs CUDA; the CPU lessons do neither.

## Why serving cares about the KV cache

At each layer, attention needs earlier tokens' keys and values. Keeping
these in a *KV cache* avoids recomputing the whole prompt for every decode
step, but a longer context costs more memory. Lesson 2's estimate showed
that cost for one standalone sequence. A server must also manage multiple
requests and allocate cache blocks. KIVI packs older cache values into 4-bit
representations, while recent values remain in a residual area. A 4-bit
value occupies half a byte before metadata and other storage costs.

Jacq's optional vLLM lane uses DSL-generated decode and cache-write CUDA
kernels in a pinned vLLM source build. That source patch is needed today
because vLLM must register the new torch operations, allocate the int4
cache, and route attention calls to them. The [setup guide](../VLLM_SETUP.md)
has the exact pinned build and generation commands. A future integration
could reduce the patch by moving the cache format and kernel dispatch behind
a stable vLLM extension interface, then upstreaming or replacing the small
registration/allocator hooks; that interface would need to support the same
cache ownership and ABI checks before the source patch can be removed.

## Read a result like an engineer

The [verified serving record](../VERIFIED_SERVING.md) reports a corrected
build on one RTX 4090, a pinned Qwen 14B AWQ model, and three arms: stock
fp16 cache, int4-KIVI with the hand-CUDA writer, and int4-KIVI with the Jacq
DSL writer. Seven deterministic completion requests matched on text, finish
reason, and token count; a chat response also matched. Instrumented calls
showed that the selected KIVI kernels actually ran. A separate direct CUDA
probe checked block-table widths at sequence boundaries. These are different
kinds of evidence:

| Check | What it establishes |
| --- | --- |
| Parse and emit on CPU | Jacq accepts a file and writes source. |
| Direct CUDA op probe | A generated operation handles specified synthetic shapes and guards. |
| Full-model serving comparison | These requests produced matching outputs on the tested build. |
| Timed workload and memory measurements | Speed and memory tradeoffs; neither follows from output agreement. |

The [portable validation bundle](../../validation/vllm-2026-11-03/README.md)
contains the requests, normalized responses, and commands for a compatible
workstation. Start with the result page, then use the [GPU validation
guide](../GPU_VALIDATION.md) if you want to repeat the experiment. The
result page owns the current measured scope; consult it before applying a
claim to another model, GPU, context length, or workload.

**Exercise:** If the Jacq DSL writer and stock fp16 return identical text
on seven prompts, does that establish the writer is faster? **Answer:** no.
Output agreement tests behavior for those requests; performance needs a
timed workload with controlled hardware and load.

## Where to go next

- Compare `--viz` before and after changing `max_seq` again; explain the
  cache estimate with the formula from lesson 2.
- Inspect `generated/gemv/gemv.cu` beside the kernel source and identify
  the host launcher and device kernel.
- On a compatible GPU, follow the validation bundle's pinned commands and
  compare operation traces as well as final text.
- Use the [ABI guide](../ABI.md) when calling a generated library kernel.
