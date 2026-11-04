// Tiled matrix-vector product (y = W·x). Run from the repository root:
//   ./target/debug/jacq --emit-cuda examples/gemv.jac
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
