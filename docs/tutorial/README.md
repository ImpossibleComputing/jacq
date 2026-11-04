# Learn Jacq, one experiment at a time

This path is for software engineers who can read code but are new to model
inference or CUDA. The first three lessons run on a CPU-only computer. You
need a Rust toolchain (including Cargo), a shell, and a text editor. No Python,
model account, download, CUDA toolkit, or GPU is needed for these lessons once
the Rust toolchain and its normal build prerequisites are available. Work through the
lessons in one shell so the temporary-directory variables carry forward.

| Lesson | Time | What you will have done |
| --- | --- | --- |
| [1. From text to generated code](01-first-compile.md) | 10 min | Parse a `.jac` file and inspect two generated files. |
| [2. Change a model setting](02-model-experiments.md) | 15 min | Edit context length and weight dtype in `.jac`, then compare generated output. |
| [3. Read a GPU kernel](03-kernel-and-memory.md) | 15 min | Edit a matrix-vector kernel and trace its data movement. |
| [4. Run and evaluate inference](04-inference-and-evidence.md) | 20 min reading; GPU time varies | Connect weights, token generation, KV caching, and the measured vLLM run. |

You can stop after lesson 3 with a working understanding of the compiler.
Lesson 4's commands require a CUDA machine and model access. The [format
guide](../JAC_FORMAT.md) is the syntax reference; the [examples
directory](../../examples/README.md) lists other runnable files.

## The map

```text
editable .jac text ──parse/lower──> model or kernel representation
                                   │
                                   └──generate──> CUDA source
                                                  + Python exporter (models)
                                                        │
                                           optional GPU build + model weights
                                                        │
                                                   generated tokens
```

Think of `.jac` as source code, the generated `.cu` file as C++/CUDA source,
and downloaded model weights as data. Reading or generating code does not
execute a model. The lessons make each boundary visible.

## Pocket glossary

- **Token:** An integer ID for a piece of text. A tokenizer turns text into
  IDs; a model predicts an ID for the next piece.
- **Inference:** Using fixed, trained weights to make predictions. Training
  changes the weights; these lessons do not train a model.
- **Tensor:** A multidimensional array. Its *shape* tells you how many values
  fit on each axis; its *dtype* tells you how each value is stored.
- **Prefill / decode:** Prefill processes the supplied prompt. Decode produces
  following tokens, commonly one step at a time.
- **Weights:** The learned numeric arrays supplied to the generated runtime.
  They are separate from the `.jac` description.
- **Kernel:** A function that runs many copies of a small operation on a GPU.
  A Jacq model file and a Jacq kernel file both use `.jac`.
- **KV cache:** Saved attention keys and values from earlier tokens so decode
  can reuse them. Lesson 4 introduces why its memory matters.
- **vLLM:** A serving runtime used by Jacq's optional int4-KIVI integration.
  It is a separate path from the standalone `jacq --run` command.
