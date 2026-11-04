# Runnable `.jac` examples

Run these commands from the repository root after `cargo build -p jac-cli`.
The compiler and emission steps need only Rust. Generated CUDA can be
inspected as text without a GPU.
For a guided introduction, start with the [tutorial](../docs/tutorial/README.md).

## Model: TinyLlama

[`tinyllama.jac`](tinyllama.jac) is a complete model spec for
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. It names a Hugging Face source for the
weight exporter; the `.jac` file contains no weights.

```bash
./target/debug/jacq --validate examples/tinyllama.jac
./target/debug/jacq --viz examples/tinyllama.jac
./target/debug/jacq --emit-model examples/tinyllama.jac
```

Emission writes `generated/tinyllama/tinyllama_inference.cu` and
`generated/tinyllama/tinyllama_export.py`.
When you have `nvcc`, an NVIDIA GPU, and Python with `torch`, `transformers`,
and `numpy`, this runs code generation, CUDA compilation, weight download and
export, then greedy inference:

```bash
./target/debug/jacq --run examples/tinyllama.jac --arch sm_89 --max-new 20
```

Use `sm_80` for an A100. The exporter writes exported weights and test tokens
under `generated/tinyllama/weights/`; that directory is ignored by Git. The
Hugging Face download may use its normal model cache. The output includes a
`REFERENCE` continuation and generated token IDs for comparison.

Save your own copy and change `max_seq`, then rerun `--validate` and `--viz` to
see how its reported configuration and memory estimate change. Changes to
architecture dimensions must still match the named weight source before
running inference.
The checked-in [short-context](tutorial/tinyllama_short.jac) and
[int8](tutorial/tinyllama_int8.jac) variants are answer references;
[lesson 2](../docs/tutorial/02-model-experiments.md) has you make both edits
yourself before inspecting the generated output.

## Kernel: GEMV

[`gemv.jac`](gemv.jac) computes a matrix-vector product. It is an example of
the kernel language, which uses `--emit-cuda` rather than `--emit-model`:

```bash
./target/debug/jacq --emit-cuda examples/gemv.jac
```

The `.cu` file includes a standalone file-IO harness. Save your own copy of
the `.jac` file, edit its dimension parameters, and re-emit to see the generated
constants change. [Lesson 3](../docs/tutorial/03-kernel-and-memory.md) shows the steps;
[gemv_small.jac](tutorial/gemv_small.jac) is the expected edited source.

For a runtime-extension example used by the opt-in vLLM integration, see
[`specs/kernels/decode_attention_kivi_paged.jac`](../specs/kernels/decode_attention_kivi_paged.jac):

```bash
./target/debug/jacq --emit-cuda --kernel-library --kivi-flash \
  specs/kernels/decode_attention_kivi_paged.jac
```

This writes `generated/decode_attention_kivi_paged/decode_attention_kivi_paged.cu`.
The callable launch ABI is documented in [ABI](../docs/ABI.md).
