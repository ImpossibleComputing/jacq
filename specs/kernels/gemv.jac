// Minimal kernel-DSL example: a tiled GEMV (y = W·x). The kernel DSL is jacq's second
// sub-language — `jacq --emit-cuda specs/gemv.jac` parses and validates it. Kernel CUDA codegen
// (turning phases into a launchable kernel, with cuBLAS/WMMA variants) is in progress.
kernel gemv {

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
