# 1. From text to generated code

**About 10 minutes.** Start in the Jacq repository root. This lesson uses
the Rust toolchain, a shell, and an editor; its commands do not use Python,
CUDA, or download weights.

An inference model takes token IDs, looks up learned numbers called *weights*,
and predicts scores for the next token. The tokenizer maps between IDs and
text. Jacq's `.jac` file describes the model's dimensions and weight source;
it is readable source text, not the weights or a runnable model.

## First success

Build the command-line compiler, then work in a fresh directory. Generated
files are written relative to your current directory, so this keeps the
checkout clean. Keep this shell open for lessons 2 and 3.

```sh
cargo build -p jac-cli
export JACQ_REPO="$(pwd)"
export JACQ_LEARN_DIR="$(mktemp -d)"
cd "$JACQ_LEARN_DIR"
"$JACQ_REPO/target/debug/jacq" --validate "$JACQ_REPO/examples/tinyllama.jac"
"$JACQ_REPO/target/debug/jacq" --viz "$JACQ_REPO/examples/tinyllama.jac"
"$JACQ_REPO/target/debug/jacq" --emit-model "$JACQ_REPO/examples/tinyllama.jac"
```

**Checkpoint:** `--validate` reports a 22-layer model with 32 query heads,
4 KV heads, and a derived `qkv_out` of 2560. `--viz` shows a pipeline and
a memory *estimate*. Emission creates `generated/tinyllama/tinyllama_inference.cu`
and `generated/tinyllama/tinyllama_export.py`. Neither file contains downloaded
model weights.

What happened? The compiler parsed the text, checked and derived dimensions
in its intermediate representation, then emitted CUDA source and a Python
weight exporter. The exporter is a program to fetch/convert weights later;
emitting it does not run it. The generated CUDA is source that a CUDA
toolchain can compile later. This is analogous to generating C++ from a
schema before compiling or running the program.

Open `$JACQ_REPO/examples/tinyllama.jac` in your editor. Its outer declaration
is `model tinyllama {`. These are its complete `config` and `weights` blocks:

```jac
config {
  hidden     = 2048
  layers     = 22
  heads      = 32
  kv_heads   = 4
  head_dim   = 64
  ffn_dim    = 5632
  vocab      = 32000
  max_seq    = 2048
  rope_base  = 10000.0
  norm_eps   = 1e-5
}

weights {
  format = huggingface
  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  dtype  = fp16
}
```

`config` supplies dimensions; `weights` names a source for the later export
step. Open `generated/tinyllama/tinyllama_inference.cu` in the same editor. Its
configuration includes this line:

```cpp
static constexpr int   MAX_SEQ         = 2048;
```

That is the connection between editable Jacq text and generated CUDA. Also
open `generated/tinyllama/tinyllama_export.py` to see the weight exporter;
you do not need to run it. Validation reads and reports, while emission writes
the two generated files.

For the complete grammar, use the [`.jac` format guide](../JAC_FORMAT.md).

Next: [change a setting and observe the result](02-model-experiments.md).
