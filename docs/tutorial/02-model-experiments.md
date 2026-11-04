# 2. Change a model setting

**About 15 minutes.** Continue in the shell and temporary directory from
[lesson 1](01-first-compile.md). You will edit two copies of TinyLlama's `.jac`
source. Checked-in [short-context](../../examples/tutorial/tinyllama_short.jac)
and [int8-weight](../../examples/tutorial/tinyllama_int8.jac) files are answer
references if you want to compare your edits afterward.

## Shapes before CUDA

A tensor's *shape* is like an array's dimensions. A token's *hidden state*
is its intermediate numeric representation inside the model. Attention
uses that state to decide which earlier positions matter: a **query** is
what the current token is looking for, **keys** offer possible matches,
and **values** hold information combined according to those matches. This
is intuition for learned vector calculations, not a literal database lookup.
A **head** is one separately learned projection for making those comparisons.

In this example, each hidden state has 2048 numbers. The 32 query heads
each have 64 numbers, so `q_dim = 32 × 64 = 2048`. Four key/value heads
each have 64, so `kv_dim = 4 × 64 = 256`. The combined Q/K/V projection
therefore has `2048 + 256 + 256 = 2560` outputs, matching `--validate`.

This is grouped-query attention (GQA): eight query heads share each KV head.
The shape affects generated buffer sizes and which weights must be loaded.
Changing architecture dimensions without matching the actual checkpoint
would not make that checkpoint fit; we will change only settings that we can
inspect safely without downloading weights.

A *dtype* is a numeric storage format. `fp16` stores a floating-point value
in 16 bits (two bytes); `int8` stores a signed integer in 8 bits (one byte).
Changing dtype can save storage but requires a rule for approximating the
original values.

## Experiment A: context length

`max_seq` is the maximum sequence length in the standalone generated model.
Open `$JACQ_REPO/examples/tinyllama.jac` and save your own copy as
`$JACQ_LEARN_DIR/tinyllama_short.jac`. In that copy, change the opening line to
`model tinyllama_short {` and make its complete `config` block read:

```jac
config {
  hidden     = 2048
  layers     = 22
  heads      = 32
  kv_heads   = 4
  head_dim   = 64
  ffn_dim    = 5632
  vocab      = 32000
  max_seq    = 256
  rope_base  = 10000.0
  norm_eps   = 1e-5
}
```

Keep the `weights`, `layer`, and `pipeline` blocks as they were. Save and
compile **your** edited file:

```sh
"$JACQ_REPO/target/debug/jacq" --validate tinyllama_short.jac
"$JACQ_REPO/target/debug/jacq" --viz tinyllama_short.jac
"$JACQ_REPO/target/debug/jacq" --emit-model tinyllama_short.jac
```

**Checkpoint:** validation says `max_seq 256`; generated
`tinyllama_short_inference.cu` has `MAX_SEQ = 256`, while the original has
`MAX_SEQ = 2048`. Open both generated `.cu` files in your editor to see their
`MAX_SEQ` lines:

```cpp
// generated/tinyllama/tinyllama_inference.cu
static constexpr int   MAX_SEQ         = 2048;

// generated/tinyllama_short/tinyllama_short_inference.cu
static constexpr int   MAX_SEQ         = 256;
```

The `--viz` fp16 KV-cache estimate falls from 44 MB to 6 MB (rounded).
There are two arrays, keys and values, at every layer:
`2 × layers × kv_heads × head_dim × max_seq × 2 bytes`. The last factor is
the size of an fp16 value. This is an estimate for one sequence, not a
measurement of total serving memory.

**Try it:** Why is the cache one eighth as large? Answer: all factors in
that estimate stayed fixed except `max_seq`, which changed from 2048 to 256.

## Experiment B: weight storage

Start again from `$JACQ_REPO/examples/tinyllama.jac` and save a separate copy
as `$JACQ_LEARN_DIR/tinyllama_int8.jac`. Change the opening line to
`model tinyllama_int8 {` and make its complete `weights` block read:

```jac
weights {
  format = huggingface
  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype  = int8
}
```

Keep the original `config` block, including `max_seq = 2048`, and leave the
`layer` and `pipeline` blocks alone. Save and compile your file:

```sh
"$JACQ_REPO/target/debug/jacq" --validate tinyllama_int8.jac
"$JACQ_REPO/target/debug/jacq" --viz tinyllama_int8.jac
"$JACQ_REPO/target/debug/jacq" --emit-model tinyllama_int8.jac
```

**Checkpoint:** open `generated/tinyllama_int8/tinyllama_int8_export.py` in
your editor; it contains `DTYPE = "int8"`. The generated
`tinyllama_int8_inference.cu` includes its int8 runtime. An `int8`
weight uses 8 bits instead of fp16's 16 bits, with quantization and scale
handling in the generated path. The `--viz` weight
estimate is explicitly fp16 and is *not* an int8 memory measurement.
Emission only changes files on disk. This lesson does not export or test
actual quantized weights.

For a small numeric example, take a weight row `[-1.0, 0.3, 1.0]`. The largest
absolute value is `1.0`, so the row's **scale** is `1/127`. Dividing by that
scale and rounding stores `[-127, 38, 127]`. Multiplying those integers by the
scale reconstructs approximately `[-1.0, 0.299213, 1.0]`. The original `0.3`
is close but not exact: this is the storage-versus-rounding tradeoff. The
generated exporter applies this rule per output row and stores the scale in
fp16, which can add a little
more rounding. Its embeddings, norms, and output head remain fp16.
This is **int8 model weight storage**; lesson 4's int4 KIVI experiment
compresses the **KV cache**, a different set of values.

The standalone model generator currently uses a supported LLaMA/Qwen-style
decoder template. `config`, `weights`, dtype, and supported LoRA options
parameterize it. The readable `layer` and `pipeline` sections appear in
inspection output; editing arbitrary expressions there does not generate a
new computational graph. For the exact syntax and supported options, see the
[format guide](../JAC_FORMAT.md).

Next: [trace one GPU kernel](03-kernel-and-memory.md).
