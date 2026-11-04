// Kernel-DSL row-wise softmax: y[r, :] = softmax(x[r, :]) over the C dimension.
// `jacq --emit-cuda specs/softmax.jac` emits a numerically-stable, one-block-per-row kernel.
kernel softmax {

  tensors {
    x[R, C]   : fp16  @global -> stream
    y[R, C]   : fp32  @registers -> global
  }

  params {
    R = 64
    C = 256
  }

  phase compute over x.dim(R) by R {
    y = softmax(x, dim: C)
  }
}
