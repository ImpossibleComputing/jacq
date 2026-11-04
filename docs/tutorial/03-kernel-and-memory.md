# 3. Read a GPU kernel

**About 15 minutes.** Continue in the shell and temporary directory from
[lesson 1](01-first-compile.md). You will generate and read CUDA, not run it.

A GPU runs many lightweight threads. Its *global memory* is large but slower
to access; *registers* are small, fast storage local to each thread. A
kernel organizes threads to reuse values and move data in tiles. The
[GEMV example](../../examples/gemv.jac) makes this visible without a whole
language model. GEMV means general matrix-vector multiplication:

```text
W has shape [M, N]    x has shape [N]    y has shape [M]
y[row] = sum over col of W[row, col] * x[col]
```

Try two rows by hand: let `W = [[1, 2, 3], [4, 5, 6]]` and `x = [2, 1, 0]`.
Then `y` is `[4, 13]`: row 0 gives `1×2 + 2×1 + 3×0 = 4`;
row 1 gives `4×2 + 5×1 + 6×0 = 13`. Each output index corresponds to one
matrix row, which is why the generated naive CUDA assigns one output row to
each GPU thread. The checked-in `.jac` file uses the same operation with
`M = 4096` output rows and `N = 128` values per row.

The `.jac` file describes `W` in global memory, register-oriented `x`/`y`,
and a phase with `tile_m = 128`. Its `einsum` line expresses the contraction.
The generated *naive GEMV* uses a concrete, simpler schedule: one GPU thread
per output row, with a float accumulator in a register. Its threads read
`W` and `x` from device memory, and the declared `tile_m` does not control
this naive schedule. Reading both files shows the difference between a
source description and its current code generator. This kernel is a useful
building block for model inference: learned weights are matrices, and a
token's hidden state is a vector. In a larger model, many projections use
related operations.

Open `$JACQ_REPO/examples/gemv.jac` in your editor. Its tensors and parameters
include:

```jac
tensors {
  x[N]      : fp16  @registers
  W[M, N]   : fp16  @global -> stream
  y[M]      : fp32  @registers -> global
}

params {
  M      = 4096
  N      = 128
  tile_m = 128
}
```

Its `phase compute` block contains the `einsum` contraction described above.
Emit the standalone CUDA source:

```sh
"$JACQ_REPO/target/debug/jacq" --emit-cuda "$JACQ_REPO/examples/gemv.jac"
```

**Checkpoint:** open `generated/gemv/gemv.cu` in your editor. It contains
`gemv_kernel` and an `int main` host harness. The harness is there to feed a
kernel during standalone execution;
`--emit-cuda` only writes source. This is a different mode from a model's
`--emit-model`, which writes a full inference runtime and weight exporter.
The generated source includes `static constexpr int M = 4096;`, and the host
launch uses `int block = 256;`. The naive emitter did not substitute
`tile_m = 128` there.

## Change a dimension

Save your own copy of `$JACQ_REPO/examples/gemv.jac` in the temporary workspace
as `gemv_small.jac`. Edit the kernel name and `M` parameter so the complete
file reads as follows:

```jac
// Tiled matrix-vector product (y = W·x).
kernel gemv_small {

  tensors {
    x[N]      : fp16  @registers
    W[M, N]   : fp16  @global -> stream
    y[M]      : fp32  @registers -> global
  }

  params {
    M      = 512
    N      = 128
    tile_m = 128
  }

  constraints {
    arch >= sm_80
    registers_per_thread <= 255
    smem_per_block <= 48KB
    occupancy >= 2
  }

  phase compute over W.dim(M) by tile_m {
    load W_tile = W[tile, *]  via cp.async
    y_tile = einsum("n, tn -> t", x, W_tile)
  }
}
```

The new name keeps the output separate from `generated/gemv/`. Save and
compile your edited source:

```sh
"$JACQ_REPO/target/debug/jacq" --emit-cuda gemv_small.jac
```

**Try it:** What shape does `y` have now? Answer: `[512]`; `N = 128` still
sets the length of `x` and each row of `W`. The generated constant now says
`M = 512`, so the naive launch covers fewer output rows. You can inspect the
resulting source in your editor without a CUDA toolkit. The generated file is
`generated/gemv_small/gemv_small.cu`, and it contains:

```cpp
static constexpr int M = 512;  // M: output rows
```

The checked-in [small GEMV](../../examples/tutorial/gemv_small.jac) is an
answer reference for these two edits.

Jacq's kernel generator recognizes specific patterns, including this GEMV.
The `.jac` declarations are a readable way to select and parameterize those
patterns; arbitrary tensor expressions are not all compiled. The
[`--kernel-library` example](../../examples/README.md) shows the separate
callable-kernel mode used by the vLLM integration.

Next: [connect generated code to actual inference](04-inference-and-evidence.md).
