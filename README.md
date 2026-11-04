# Jacq Inference

Jacq turns readable `.jac` model and kernel descriptions into CUDA. Write a
model specification, inspect what the compiler understands, and generate a
standalone inference runtime and weight exporter. Kernel specifications can
generate standalone CUDA harnesses or supported callable library kernels.

**New here?** A `.jac` file is text, not a model checkpoint. Follow the
[four-lesson tutorial](docs/tutorial/README.md) for a CPU-only first success
and a path to GPU inference. If you already know the basics, use the
[five-minute quick start](docs/QUICK_START.md), [format guide](docs/JAC_FORMAT.md),
or [runnable examples](examples/README.md).

## First run (Rust only)

From the repository root:

```bash
cargo build -p jac-cli
./target/debug/jacq --validate examples/tinyllama.jac
./target/debug/jacq --viz examples/tinyllama.jac
./target/debug/jacq --emit-model examples/tinyllama.jac
ls generated/tinyllama/
```

To install the `jacq` command on your PATH, run `cargo install --path crates/jac-cli`.
If you installed this checkout when the command was `jac`, Cargo leaves that
old executable alongside `jacq`. Remove it with `cargo uninstall --bin jac jac-cli`.

You should see `tinyllama_inference.cu` and `tinyllama_export.py`. These
commands need no GPU or model download. The `.jac` file names the Hugging Face
weight source; `--emit-model` generates the exporter but does not run it.
To compile CUDA, export weights, and generate tokens on an NVIDIA GPU, follow
the [next step in Quick start](docs/QUICK_START.md).

## What is in this repository

- `examples/` — complete `.jac` model and kernel files with commands to run.
- `docs/tutorial/` — worked lessons from source text through serving evidence.
- `crates/` — parser, model IR, CUDA code generation, and the `jacq` CLI.
- `specs/models/` and `specs/kernels/` — further model and kernel definitions.
- `integrations/vllm_kivi/` — an opt-in paged int4-KIVI integration for a
  pinned vLLM source checkout, with a versioned callable CUDA interface.
- `tests/` — Python tests for the vLLM integration and ABI checks.

The standalone `--run` path builds a generated CUDA runtime and exports
weights for local inference. The [vLLM setup guide](docs/VLLM_SETUP.md) covers
the separate source integration for serving experiments. See
[ABI](docs/ABI.md) for its launch contract and
[GPU validation](docs/GPU_VALIDATION.md) for a reproducible serving check.

## Verification

The compiler and integration checks run on a CPU-only machine with
`cargo test --workspace` and `python3 -m unittest discover -s tests -v`.
An RTX 4090 direct-launch smoke of the emitted paged KIVI kernel passed four
synthetic cases. A [verified vLLM serving run](docs/VERIFIED_SERVING.md) on
November 3, 2026 matched stock fp16 output across seven completion requests
for both int4-KIVI cache writers on one 14B model. See
[the A100 serving follow-up](docs/VERIFIED_SERVING.md#a100-follow-up-november-4-2026)
for a second GPU architecture tested with the public Jacq release and a
smaller synthetic request set. See
[the model-family checks](validation/model-families-2026-11-03/README.md) for
scoped standalone TinyLlama, Qwen2.5, and Mistral results and
[the startup compatibility follow-up](validation/model-families-2026-11-04/README.md)
for early rejection of unsupported KIVI models. See
[GPU validation](docs/GPU_VALIDATION.md) for the repeatable procedure and
[current boundaries](docs/LIMITATIONS.md) for the supported lane.

Jacq is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions
and local test commands are in [CONTRIBUTING.md](CONTRIBUTING.md).
