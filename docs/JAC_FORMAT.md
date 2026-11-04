# The `.jac` file format

A `.jac` file is **plain text source code** for Jacq. You can write one in any
text editor, keep it in Git, and read it without a special viewer. The file
describes *what* model or kernel to build; the `jacq` command reads it and
generates CUDA and, for models, a Python weight exporter. A `.jac` file does
not contain model weights or a compiled binary.

Start with [the five-minute example](QUICK_START.md), then open
[`examples/tinyllama.jac`](../examples/tinyllama.jac) alongside this guide.

## A model file

This shortened excerpt shows the main pieces:

```jac
model tinyllama {
  config {
    hidden = 2048
    layers = 22
    heads = 32
    kv_heads = 4
    head_dim = 64
    vocab = 32000
  }

  weights {
    format = huggingface
    source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype = fp16
  }

  layer decoder {
    normed = rmsnorm(input, norm_weight, norm_eps)
    q, k, v = linear(normed, w_q, w_k, w_v)
    // See the complete example for attention and feed-forward steps.
  }

  pipeline {
    embed(token, embed_weights)
    -> repeat(decoder, layers)
    -> greedy_sample
  }
}
```

This excerpt illustrates the shape of a file; use the complete example for
commands. Each model file contains one `model name { ... }` declaration:

| Section | Meaning |
| --- | --- |
| `config` | Model dimensions and settings. `hidden`, `layers`, `heads`, `head_dim`, and `vocab` are required. `kv_heads` defaults to `heads`; `ffn_dim`, `max_seq`, `rope_base`, and `norm_eps` have compiler defaults. |
| `weights` | Where the exporter should get weights and which output dtype to use. In this example, the quoted `source` is a Hugging Face model ID; a local directory supported by Transformers can also be used. |
| `layer` | Named assignments that express the intended decoder block. The name is referenced by `repeat(...)` in the pipeline. |
| `pipeline` | A readable description of embedding, repeated layers, final projection, and sampling. |

The current standalone emitter generates a supported LLaMA/Qwen-style
decoder template. Configuration, weights, dtype, and optional LoRA settings
parameterize that template. `layer` and `pipeline` appear in inspection
output, but changing their expressions or order does not create a new
computational graph in the generated runtime.

The `model` and `kernel` languages share the `.jac` extension, but their
commands are different. `jacq --validate` and `jacq --emit-model` accept a model
file. `jacq --emit-cuda` accepts a kernel file. The CLI reads one declaration
per file.

Comments begin with `//`. Names and enum-like values such as `fp16` are bare
words; text such as a model ID is quoted. Assignments use `=`, blocks use
braces, and pipeline steps use `->`. Whitespace and indentation are for the
reader. Numbers can be integers (`2048`) or floats (`1e-5`, `10000.0`).

From the repository root, try:

```bash
cargo build -p jac-cli
./target/debug/jacq --validate examples/tinyllama.jac
./target/debug/jacq --viz examples/tinyllama.jac
./target/debug/jacq --emit-model examples/tinyllama.jac
```

The first two commands read the file without creating output. `--emit-model`
writes `generated/tinyllama/tinyllama_inference.cu` and
`generated/tinyllama/tinyllama_export.py` relative to your **current working
directory**. It does not download weights or run inference. For that next
step, see [Quick start](QUICK_START.md).

## A kernel file

[`examples/gemv.jac`](../examples/gemv.jac) is a small matrix-vector product.
It declares input and output tensors, concrete parameters, device constraints,
and a `phase` that loads a tile and contracts it with `einsum`:

```jac
kernel gemv {
  tensors {
    x[N]    : fp16 @registers
    W[M, N] : fp16 @global -> stream
    y[M]    : fp32 @registers -> global
  }
  params { M = 4096 N = 128 tile_m = 128 }
  phase compute over W.dim(M) by tile_m {
    load W_tile = W[tile, *] via cp.async
    y_tile = einsum("n, tn -> t", x, W_tile)
  }
}
```

Run the complete checked-in file:

```bash
./target/debug/jacq --emit-cuda examples/gemv.jac
```

That writes `generated/gemv/gemv.cu`, a CUDA kernel with a standalone host
harness. Kernel code generation recognizes specific patterns (including GEMV,
matmul, row-wise softmax, and selected attention forms); the declarations
alone do not make every possible tensor expression compilable. The
`--kernel-library` option is for supported runtime-extension kernels and
emits a translation unit without the standalone harness. See
[the kernel walkthrough](../examples/README.md) for the exact command.
To change a kernel parameter yourself, copy the file to a working directory,
edit its `params` in a text editor, and re-emit it. [Lesson 3](tutorial/03-kernel-and-memory.md)
shows the complete edit and expected output.

## Choosing where to go next

- [Quick start](QUICK_START.md): validate and emit your first `.jac` file.
- [Examples](../examples/README.md): copy and modify a model or kernel.
- [vLLM setup](VLLM_SETUP.md): use the opt-in paged int4-KIVI integration.
- [ABI](ABI.md): details for callers of the generated library kernels.
