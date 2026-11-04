# Quick start

This first run needs a Rust toolchain (including Cargo), a shell, and a text
editor; it needs no Python, CUDA toolkit, or GPU. For explanation and exercises,
follow the [beginner tutorial](tutorial/README.md). Work from the repository
root so the example paths and generated output paths match the commands below.

## 1. Build and read a `.jac` file

```bash
cargo build -p jac-cli
./target/debug/jacq --version
```

Open [`examples/tinyllama.jac`](../examples/tinyllama.jac). It is plain text
with four parts: `config` gives the model dimensions, `weights` names a
Hugging Face source and output dtype, and `layer` and `pipeline` show the
intended decoder flow. The source name is a reference for the later
weight-export step. There are no weights in the `.jac` file.

## 2. Validate and inspect it

```bash
./target/debug/jacq --validate examples/tinyllama.jac
./target/debug/jacq --viz examples/tinyllama.jac
```

`--validate` parses and lowers the model and reports its derived dimensions.
`--viz` prints the configuration, pipeline, and a memory estimate. Neither
command downloads a model or writes generated files. To view an interactive
HTML diagram, `./target/debug/jacq --viz-web examples/tinyllama.jac` writes
`tinyllama_viz.html` in the current directory.

## 3. Generate the runtime

```bash
./target/debug/jacq --emit-model examples/tinyllama.jac
```

Expect `tinyllama_inference.cu` (standalone CUDA inference runtime) and
`tinyllama_export.py` (Python weight exporter). `generated/` is created in
the current working directory and is ignored by Git. Emission does not run
CUDA, download weights, or require an NVIDIA GPU.

## 4. Run inference on a GPU

With `nvcc`, an NVIDIA GPU, and Python 3 with `torch`, `transformers`, and
`numpy` installed, run:

```bash
./target/debug/jacq --doctor
./target/debug/jacq --run examples/tinyllama.jac --arch sm_89 --max-new 20
```

Use `sm_80` for an A100. `--run` emits the same files, compiles the `.cu`,
downloads the model through Hugging Face and writes exported weights into
`generated/tinyllama/weights/`,
then runs greedy inference on test tokens. It prints reference and generated
token IDs so you can compare them. Downloading the named model requires
network access and sufficient disk space.

## Try a kernel file

Model and kernel files both end in `.jac`, but use different compiler modes.
The complete [`examples/gemv.jac`](../examples/gemv.jac) describes a small
matrix-vector kernel:

```bash
./target/debug/jacq --emit-cuda examples/gemv.jac
```

`--emit-cuda` writes a standalone CUDA translation unit and host harness.
For supported runtime-extension kernels, `--kernel-library` writes a callable
translation unit instead; see [Examples](../examples/README.md).

Next, read [the `.jac` format guide](JAC_FORMAT.md) or take the
[worked model and kernel editing lessons](tutorial/README.md).
For the opt-in vLLM integration, continue to [vLLM setup](VLLM_SETUP.md).
