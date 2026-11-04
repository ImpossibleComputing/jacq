// Kernel-DSL matmul: C = A·B. `jacq --emit-cuda specs/matmul.jac` emits a naive kernel;
// `--cublas` delegates to cublasGemmEx. Both compute row-major C[M,K] = A[M,N] · B[N,K].
kernel matmul {

  tensors {
    A[M, N]   : fp16  @global -> stream
    B[N, K]   : fp16  @global -> stream
    C[M, K]   : fp32  @registers -> global
  }

  params {
    M      = 256
    N      = 128
    K      = 64
    tile_m = 64
  }

  phase compute over A.dim(M) by tile_m {
    load A_tile = A[tile, *]
    C_tile = einsum("mn, nk -> mk", A_tile, B)
  }
}
