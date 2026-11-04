//! CUDA variant emitters for the kernel DSL: GEMV, matmul, and row-wise softmax.
//!
//! Each operation has three layers:
//! - A `*_device_kernel(name)` helper returning the bare `__global__` function source (ending
//!   with `}\n\n`), shared by both the standalone `emit_*` (with `main()`) and the preamble-free
//!   `emit_*_body` (with `jac_launch`) paths.
//! - An `emit_*_body` function used by the experiment harness via `emit_kernel_body`.
//! - An `emit_*` function used by the standalone `emit_kernel` path.

use jac_syntax::ast::{KStmt, Kernel};

/// Which CUDA backend to emit for a recognized kernel. Strategy level 1: the user selects the
/// variant (e.g. `jacq --emit-cuda --cublas`); the spec is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelVariant {
    /// A hand-written naive kernel (one thread per output row, fp32 accumulate).
    #[default]
    Naive,
    /// Delegate the contraction to cuBLAS (`cublasGemmEx`).
    Cublas,
    /// Tensor-core (`nvcuda::wmma`) matmul: smem-staged, one warp per 16×16 output tile, fp16
    /// inputs / fp32 accumulate. Step 1 of the WMMA roadmap; step 2 (`WmmaOpt`) is additive.
    Wmma,
    /// KIVI-only flash-decode schedule: one warp per query head, online softmax, and
    /// vectorized int4 `load4` dequant, mirroring the reference hand-written CUDA schedule.
    /// Split-K and GQA-amortized variants remain separate future lowerings.
    KiviFlash,
    /// Generic (non-KIVI) paged flash-decode schedule: one warp per query head, online softmax,
    /// arbitrary GQA ratio, uniformly-fp16 K/V gathered from pages via `BLOCK_TABLE`. The
    /// non-quantized sibling of `KiviFlash` -- no dequant, no scale/zero, no residual split.
    PagedFlash,
}

// ---- WMMA config ----------------------------------------------------------------

/// Configuration for the config-driven WMMA matmul emitter.
///
/// Step 1 (trivial instance via [`wmma_step1_config`]): one warp computes a single 16×16 output
/// tile; K-loop in steps of 16; single-buffered shared memory. All tunables that step 2
/// (`wmma_opt`) will vary are already named here so the emitter helpers are config-keyed and the
/// step-2 upgrade is purely additive.
pub(crate) struct WmmaConfig {
    /// Output tile rows per thread-block.
    pub(crate) block_tile_m: u32,
    /// Output tile cols per thread-block.
    pub(crate) block_tile_n: u32,
    /// Output tile rows per warp (step 1: same as block tile; step 2: multiple of 16).
    /// Reserved for step-2 warp-tiling; unused in the step-1 emitter.
    #[allow(dead_code)]
    pub(crate) warp_tile_m: u32,
    /// Output tile cols per warp (step 1: same as block tile; step 2: multiple of 16).
    /// Reserved for step-2 warp-tiling; unused in the step-1 emitter.
    #[allow(dead_code)]
    pub(crate) warp_tile_n: u32,
    /// K advanced per main-loop iteration (must be 16 for nvcuda::wmma).
    pub(crate) k_step: u32,
    /// Warps per thread-block (step 1: 1; step 2: 4–8).
    /// Reserved for step-2 multi-warp blocks; unused in the step-1 emitter.
    #[allow(dead_code)]
    pub(crate) warps_per_block: u32,
    /// 1 = single-buffered (step 1); 2 = double-buffered cp.async (step 2).
    pub(crate) pipeline_stages: u32,
}

/// The trivial step-1 config: one warp, 16×16 tile, k_step = 16, single-buffered.
pub(crate) fn wmma_step1_config() -> WmmaConfig {
    WmmaConfig {
        block_tile_m: 16,
        block_tile_n: 16,
        warp_tile_m: 16,
        warp_tile_n: 16,
        k_step: 16,
        warps_per_block: 1,
        pipeline_stages: 1,
    }
}

// ---- WMMA emitter helpers -------------------------------------------------------
// These helpers are the shared substrate for step 1 and step 2 (wmma_opt).  Step 2 calls the same
// helpers and adds `emit_kloop_pipelined` beside `emit_kloop_simple`; nothing here is reworked.

/// Emit `wmma::fragment` declarations and the `fill_fragment` zero-initialisation for the
/// fp32 accumulator.  Both step 1 and step 2 use this unchanged.
pub(crate) fn emit_wmma_fragments(name: &str) -> String {
    format!(
        "    // WMMA fragment declarations — shared substrate for all pipeline stages.\n\
         \x20   wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;\n\
         \x20   wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_b;\n\
         \x20   wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_acc;\n\
         \x20   wmma::fill_fragment(frag_acc, 0.0f);  // zero accumulator for kernel `{name}`\n"
    )
}

/// Emit the single-buffered K-loop body (load fragments directly from global, `mma_sync`).
/// Keyed on `pipeline_stages == 1`.  Step 2 adds `emit_kloop_pipelined` (stages == 2) beside this
/// function without touching it — the seam is the match in `emit_matmul_wmma`.
///
/// Step 1 loads fragments directly from global memory (no shared-memory staging) — the simplest
/// correct WMMA kernel.  Shared-memory tiling for data reuse is a step-2 (`wmma_opt`) concern.
/// The smem staging in the previous approach was incorrect: a 32-thread warp cannot cooperatively
/// fill a 16×16 = 256-element tile with one element per thread (only 32/256 elements would be
/// written; the rest stay uninitialized, producing rel_err ≈ 25).  Additionally the old loop
/// bound used `K` (the output-column count) instead of `N` (the contraction length).
pub(crate) fn emit_kloop_simple(cfg: &WmmaConfig) -> String {
    format!(
        "    // K-loop: step over the contraction axis (jacq-N) in tiles of {ks} (pipeline_stages={ps}).\n\
         \x20   int warp_row = (blockIdx.y * {btm_div16}) * 16;  // first output row of this warp (M-axis)\n\
         \x20   int warp_col = (blockIdx.x * {btn_div16}) * 16;  // first output col of this warp (K-axis)\n\
         \x20   for (int k_base = 0; k_base < N; k_base += {ks}) {{\n\
         \x20       // Load A tile: A[warp_row .. +16][k_base .. +16], row-major ld = N.\n\
         \x20       wmma::load_matrix_sync(frag_a, A + (size_t)warp_row * N + k_base, N);\n\
         \x20       // Load B tile: B[k_base .. +16][warp_col .. +16], row-major ld = K.\n\
         \x20       wmma::load_matrix_sync(frag_b, B + (size_t)k_base * K + warp_col, K);\n\
         \x20       wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);\n\
         \x20   }}\n",
        ps = cfg.pipeline_stages,
        btm_div16 = cfg.block_tile_m / 16,
        btn_div16 = cfg.block_tile_n / 16,
        ks = cfg.k_step,
    )
}

/// Emit `store_matrix_sync` that writes the fp32 accumulator fragment directly to C (row-major,
/// ld = K).  Both step 1 and step 2 use this unchanged.
///
/// Step 1 stores directly to global memory — no smem_C staging needed.  The smem_C approach had
/// the same 32-thread / 256-element mismatch as smem_A/B: only 32 of 256 elements would be
/// scattered back to C, silently zeroing the rest.
pub(crate) fn emit_wmma_store(_cfg: &WmmaConfig) -> String {
    // Step 1: store directly to C global — warp_row/warp_col/K are in scope from emit_kloop_simple.
    // Step 2 will use _cfg to parameterise the store (e.g. multiple 16×16 tiles per warp).
    "    // Store accumulator directly to C[warp_row..+16][warp_col..+16], row-major ld = K.\n\
     \x20   wmma::store_matrix_sync(C + (size_t)warp_row * K + warp_col, frag_acc, K, wmma::mem_row_major);\n"
        .to_string()
}

/// The `__global__` device kernel for the WMMA variant.  No preamble; no host code.
/// Returns the kernel function ending with `}\n\n`.
///
/// Step 1: fragments declared → accumulator zeroed → K-loop (direct global loads) → store to C.
/// Step 2 (`wmma_opt`) will re-introduce smem staging via a separate `emit_smem_tile_layout` call
/// alongside `emit_kloop_pipelined`; this function gains a `cfg.pipeline_stages == 2` arm then.
pub(crate) fn wmma_device_kernel(name: &str, cfg: &WmmaConfig) -> String {
    let fragments = emit_wmma_fragments(name);
    let kloop = match cfg.pipeline_stages {
        1 => emit_kloop_simple(cfg),
        // Step 2 adds: 2 => emit_kloop_pipelined(cfg),
        _ => panic!("unsupported pipeline_stages {}", cfg.pipeline_stages),
    };
    let store = emit_wmma_store(cfg);
    format!(
        "__global__ void {name}_kernel(const __half* A, const __half* B, float* C) {{\n\
         {fragments}\
         {kloop}\
         {store}\
         }}\n\n"
    )
}

/// Emit the standalone `.cu` (device kernel + `main()` harness) for the WMMA matmul variant.
/// Top-level composer: `emit_wmma_fragments` + `emit_kloop_*` + `emit_wmma_store` →
/// `__global__` kernel + standalone host harness.
pub(crate) fn emit_matmul_wmma(k: &Kernel, mm: &Matmul, cfg: &WmmaConfig) -> String {
    let name = &k.name;
    let kernel_def = wmma_device_kernel(name, cfg);
    let compute = format!(
        "    // Grid: (N/16) blocks in x (cols), (M/16) blocks in y (rows); 32 threads/block (1 warp).\n\
         \x20   dim3 grid((K + 15) / 16, (M + 15) / 16);\n\
         \x20   dim3 block(32);  // exactly one warp\n\
         \x20   {name}_kernel<<<grid, block>>>(dA, dB, dC);\n\
         \x20   CUDA_CHECK(cudaGetLastError());"
    );
    format!(
        r#"// Generated by jacq — kernel `{name}` (matmul: C[{m},{kk}] = sum_n A[{m},{nn}] * B[{nn},{kk}]).
// WMMA tensor-core variant (nvcuda::wmma, fp16 in / fp32 accumulate, direct global loads).
// One warp per 16x16 output tile; grid = (K/16) x (M/16). Regenerate; do not hand-edit.
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using namespace nvcuda;

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int M = {m_size};  // {m}
static constexpr int N = {n_size};  // {nn}: contraction
static constexpr int K = {k_size};  // {kk}

{kernel_def}static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 4) {{
        fprintf(stderr, "usage: %s <A.bin [M*N fp16]> <B.bin [N*K fp16]> <C.bin [M*K fp32 out]>\n", argv[0]);
        return 1;
    }}
    __half* dA = load_half(argv[1], (size_t)M * N);
    __half* dB = load_half(argv[2], (size_t)N * K);
    float* dC;
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * K * sizeof(float)));

{compute}

    float* hC = (float*)malloc((size_t)M * K * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[3], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }}
    fwrite(hC, sizeof(float), (size_t)M * K, out);
    fclose(out);
    printf("C[0:8] =");
    for (int i = 0; i < 8 && i < M * K; ++i) printf(" %.4f", hC[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        m = mm.m_dim,
        nn = mm.n_dim,
        kk = mm.k_dim,
        m_size = mm.m,
        n_size = mm.n,
        k_size = mm.k,
        kernel_def = kernel_def,
        compute = compute,
    )
}

/// Emit the preamble-free body (device kernel + `jac_launch`) for the WMMA matmul variant.
/// The including harness template provides `#include <cuda_fp16.h>`, `CUDA_CHECK`, and dim macros;
/// this function emits ONLY the kernel + launch wrapper (mirror of `emit_matmul_body`).
pub(crate) fn emit_matmul_wmma_body(k: &Kernel, _mm: &Matmul, cfg: &WmmaConfig) -> String {
    let name = &k.name;
    let kernel = wmma_device_kernel(name, cfg);
    format!(
        "#include <mma.h>\n\
         using namespace nvcuda;\n\
         {kernel}\
         // jac_launch: inputs in tensor-declaration order (A[M,N], B[N,K]), then output C[M,K], then stream.\n\
         extern \"C\" void jac_launch(const __half* A, const __half* B, float* C, cudaStream_t stream) {{\n\
         \x20   dim3 grid((K + 15) / 16, (M + 15) / 16);\n\
         \x20   dim3 block(32);  // exactly one warp\n\
         \x20   {name}_kernel<<<grid, block, 0, stream>>>(A, B, C);\n\
         \x20   CUDA_CHECK(cudaGetLastError());\n\
         }}\n",
    )
}

/// Validate that M, N, K are all multiples of 16 (required by the 16×16×16 WMMA tile).
/// Returns `Err` with a descriptive message if any dimension fails.
pub(crate) fn check_wmma_dims(mm: &Matmul) -> Result<(), String> {
    let check = |name: &str, v: i64| {
        if v % 16 != 0 {
            Err(format!(
                "wmma requires M/N/K multiples of 16; got {name}={v} (not a multiple of 16)"
            ))
        } else {
            Ok(())
        }
    };
    check(&mm.m_dim, mm.m)?;
    check(&mm.n_dim, mm.n)?;
    check(&mm.k_dim, mm.k)?;
    Ok(())
}

// ---- GEMV -----------------------------------------------------------------------

/// The pieces of a recognized GEMV kernel.
pub(crate) struct Gemv {
    pub(crate) out_dim: String,
    pub(crate) out_size: i64,
    pub(crate) contract_dim: String,
    pub(crate) contract_size: i64,
}

/// Recognize the matrix-vector pattern: a 1-D output tensor (`-> global`), a 1-D input vector, a
/// 2-D `@global` matrix over `[out, contract]`, and a two-operand matrix-vector einsum.
pub(crate) fn match_gemv(k: &Kernel) -> Option<Gemv> {
    let output = k.tensors.iter().find(|t| t.output && t.dims.len() == 1)?;
    let out_dim = output.dims[0].clone();
    let vec_in = k
        .tensors
        .iter()
        .find(|t| !t.output && t.dims.len() == 1 && t.name != output.name)?;
    let contract_dim = vec_in.dims[0].clone();
    let mat = k.tensors.iter().find(|t| t.dims.len() == 2)?;
    if mat.dims != [out_dim.clone(), contract_dim.clone()] {
        return None;
    }
    if !is_matvec_subscripts(einsum_subscripts(k)?) {
        return None;
    }
    Some(Gemv {
        out_size: param(k, &out_dim)?,
        contract_size: param(k, &contract_dim)?,
        out_dim,
        contract_dim,
    })
}

/// Shared `__global__` device-kernel source for GEMV (Naive variant).
/// Returns the kernel function ending with `}\n\n`; no preamble, no host code.
pub(crate) fn gemv_device_kernel(name: &str) -> String {
    format!(
        "__global__ void {name}_kernel(const __half* W, const __half* x, float* y) {{\n\
         \x20   int m = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (m >= M) return;\n\
         \x20   float acc = 0.0f;\n\
         \x20   for (int n = 0; n < N; ++n)\n\
         \x20       acc += __half2float(W[(size_t)m * N + n]) * __half2float(x[n]);\n\
         \x20   y[m] = acc;\n\
         }}\n\n"
    )
}

/// Emit just the device kernel(s) and a stream-aware `jac_launch` wrapper for GEMV.
/// Parameter order for `jac_launch`: non-output tensors in declaration order (x, W), then y, then
/// stream.
pub(crate) fn emit_gemv_body(k: &Kernel, _g: &Gemv, variant: KernelVariant) -> String {
    let name = &k.name;
    match variant {
        KernelVariant::Naive => format!(
            "{kernel}\
             // jac_launch: inputs in tensor-declaration order (x[N], W[M,N]), then output y[M], then stream.\n\
             extern \"C\" void jac_launch(const __half* x, const __half* W, float* y, cudaStream_t stream) {{\n\
             \x20   int block = 256;\n\
             \x20   {name}_kernel<<<(M + block - 1) / block, block, 0, stream>>>(W, x, y);\n\
             \x20   CUDA_CHECK(cudaGetLastError());\n\
             }}\n",
            kernel = gemv_device_kernel(name),
        ),
        KernelVariant::Cublas =>
            "// jac_launch: inputs in tensor-declaration order (x[N], W[M,N]), then output y[M], then stream.\n\
             extern \"C\" void jac_launch(const __half* x, const __half* W, float* y, cudaStream_t stream) {\n\
              \x20  cublasHandle_t handle;\n\
              \x20  cublasCreate(&handle);\n\
              \x20  cublasSetStream(handle, stream);\n\
              \x20  const float alpha = 1.0f, beta = 0.0f;\n\
              \x20  // y[M] = W[M,N] @ x[N], W row-major: OP_T over a column-major [N,M] view (lda=N).\n\
              \x20  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N,\n\
              \x20               &alpha, W, CUDA_R_16F, N, x, CUDA_R_16F, N,\n\
              \x20               &beta, y, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);\n\
              \x20  CUDA_CHECK(cudaGetLastError());\n\
              \x20  cublasDestroy(handle);\n\
             }\n"
                .to_string(),
        // WMMA is a matmul-only variant; the caller routes GEMV to a different variant.
        KernelVariant::Wmma => unreachable!("Wmma variant does not support GEMV"),
        KernelVariant::KiviFlash => unreachable!("KiviFlash variant does not support GEMV"),
        KernelVariant::PagedFlash => unreachable!("PagedFlash variant does not support GEMV"),
    }
}

pub(crate) fn emit_gemv(k: &Kernel, g: &Gemv, variant: KernelVariant) -> String {
    let name = &k.name;
    // Variant-specific pieces: header note, extra includes, an optional kernel definition, and the
    // compute block inside main(). The host harness (I/O, buffers) is shared.
    let (note, includes, kernel_def, compute) = match variant {
        KernelVariant::Naive => (
            "Naive one-thread-per-output-row kernel with fp32 accumulation.",
            "",
            gemv_device_kernel(name),
            format!(
                "    int block = 256;\n\
                 \x20   {name}_kernel<<<(M + block - 1) / block, block>>>(dW, dx, dy);\n\
                 \x20   CUDA_CHECK(cudaGetLastError());"
            ),
        ),
        KernelVariant::Cublas => (
            "cuBLAS-delegated contraction (cublasGemmEx, fp16 in / fp32 out).",
            "#include <cublas_v2.h>\n",
            String::new(),
            // y[M] = W[M,N] @ x[N], W row-major: OP_T over a column-major [N,M] view (lda=N).
            "    cublasHandle_t handle;\n\
             \x20   cublasCreate(&handle);\n\
             \x20   const float alpha = 1.0f, beta = 0.0f;\n\
             \x20   cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N,\n\
             \x20                &alpha, dW, CUDA_R_16F, N, dx, CUDA_R_16F, N,\n\
             \x20                &beta, dy, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);\n\
             \x20   CUDA_CHECK(cudaGetLastError());\n\
             \x20   cublasDestroy(handle);"
                .to_string(),
        ),
        // WMMA is a matmul-only variant; the caller routes GEMV to a different variant.
        KernelVariant::Wmma => unreachable!("Wmma variant does not support GEMV"),
        KernelVariant::KiviFlash => unreachable!("KiviFlash variant does not support GEMV"),
        KernelVariant::PagedFlash => unreachable!("PagedFlash variant does not support GEMV"),
    };
    format!(
        r#"// Generated by jacq — kernel `{name}` (GEMV: y[{m}] = sum_n x[n] * W[{m},{n}]).
// {note} Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
{includes}#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int M = {m_size};  // {m}: output rows
static constexpr int N = {n_size};  // {n}: contraction length

{kernel_def}// Read `count` fp16 values from `path` into a fresh device buffer.
static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 4) {{
        fprintf(stderr, "usage: %s <x.bin [N fp16]> <W.bin [M*N fp16]> <y.bin [M fp32 out]>\n", argv[0]);
        return 1;
    }}
    __half* dx = load_half(argv[1], (size_t)N);
    __half* dW = load_half(argv[2], (size_t)M * N);
    float* dy;
    CUDA_CHECK(cudaMalloc(&dy, (size_t)M * sizeof(float)));

{compute}

    float* hy = (float*)malloc((size_t)M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hy, dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[3], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }}
    fwrite(hy, sizeof(float), (size_t)M, out);
    fclose(out);
    printf("y[0:8] =");
    for (int i = 0; i < 8 && i < M; ++i) printf(" %.4f", hy[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        note = note,
        includes = includes,
        kernel_def = kernel_def,
        compute = compute,
        m = g.out_dim,
        n = g.contract_dim,
        m_size = g.out_size,
        n_size = g.contract_size,
    )
}

// ---- Matmul ---------------------------------------------------------------------

/// The pieces of a recognized matmul kernel: `C[m, k] = Σ_n A[m, n] · B[n, k]`.
pub(crate) struct Matmul {
    pub(crate) m_dim: String,
    pub(crate) n_dim: String,
    pub(crate) k_dim: String,
    pub(crate) m: i64,
    pub(crate) n: i64,
    pub(crate) k: i64,
}

/// Recognize the matrix-matrix pattern: a 2-D output `C[M, K]` (`-> global`), inputs `A[M, N]`
/// and `B[N, K]`, and a two-operand matmul einsum (`ab, bc -> ac`).
pub(crate) fn match_matmul(k: &Kernel) -> Option<Matmul> {
    let out = k.tensors.iter().find(|t| t.output && t.dims.len() == 2)?;
    let (m_dim, k_dim) = (out.dims[0].clone(), out.dims[1].clone());
    // A shares the output's row dim M; B shares its column dim K; they meet on the contraction N.
    let a = k
        .tensors
        .iter()
        .find(|t| !t.output && t.dims.len() == 2 && t.dims[0] == m_dim && t.name != out.name)?;
    let n_dim = a.dims[1].clone();
    let b = k.tensors.iter().find(|t| {
        !t.output
            && t.dims.len() == 2
            && t.name != out.name
            && t.name != a.name
            && t.dims == [n_dim.clone(), k_dim.clone()]
    })?;
    let _ = b;
    if !is_matmul_subscripts(einsum_subscripts(k)?) {
        return None;
    }
    Some(Matmul {
        m: param(k, &m_dim)?,
        n: param(k, &n_dim)?,
        k: param(k, &k_dim)?,
        m_dim,
        n_dim,
        k_dim,
    })
}

/// Shared `__global__` device-kernel source for matmul (Naive variant).
/// Returns the kernel function ending with `}\n\n`; no preamble, no host code.
pub(crate) fn matmul_device_kernel(name: &str) -> String {
    format!(
        "__global__ void {name}_kernel(const __half* A, const __half* B, float* C) {{\n\
         \x20   int row = blockIdx.y * blockDim.y + threadIdx.y;  // m\n\
         \x20   int col = blockIdx.x * blockDim.x + threadIdx.x;  // k\n\
         \x20   if (row >= M || col >= K) return;\n\
         \x20   float acc = 0.0f;\n\
         \x20   for (int n = 0; n < N; ++n)\n\
         \x20       acc += __half2float(A[(size_t)row * N + n]) * __half2float(B[(size_t)n * K + col]);\n\
         \x20   C[(size_t)row * K + col] = acc;\n\
         }}\n\n"
    )
}

/// Emit just the device kernel(s) and a stream-aware `jac_launch` wrapper for matmul.
/// Parameter order for `jac_launch`: non-output tensors in declaration order (A, B), then C, then
/// stream.
pub(crate) fn emit_matmul_body(k: &Kernel, _mm: &Matmul, variant: KernelVariant) -> String {
    let name = &k.name;
    match variant {
        KernelVariant::Naive => format!(
            "{kernel}\
             // jac_launch: inputs in tensor-declaration order (A[M,N], B[N,K]), then output C[M,K], then stream.\n\
             extern \"C\" void jac_launch(const __half* A, const __half* B, float* C, cudaStream_t stream) {{\n\
             \x20   dim3 block(16, 16);\n\
             \x20   dim3 grid((K + 15) / 16, (M + 15) / 16);\n\
             \x20   {name}_kernel<<<grid, block, 0, stream>>>(A, B, C);\n\
             \x20   CUDA_CHECK(cudaGetLastError());\n\
             }}\n",
            kernel = matmul_device_kernel(name),
        ),
        KernelVariant::Cublas =>
            "// jac_launch: inputs in tensor-declaration order (A[M,N], B[N,K]), then output C[M,K], then stream.\n\
             extern \"C\" void jac_launch(const __half* A, const __half* B, float* C, cudaStream_t stream) {\n\
              \x20  cublasHandle_t handle;\n\
              \x20  cublasCreate(&handle);\n\
              \x20  cublasSetStream(handle, stream);\n\
              \x20  const float alpha = 1.0f, beta = 0.0f;\n\
              \x20  // Row-major C[M,K]=A[M,N]*B[N,K] via col-major Ct=Bt*At: GemmEx(N,N,K,M,N,B(ld=K),A(ld=N),C(ld=K)).\n\
              \x20  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,\n\
              \x20               &alpha, B, CUDA_R_16F, K, A, CUDA_R_16F, N,\n\
              \x20               &beta, C, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);\n\
              \x20  CUDA_CHECK(cudaGetLastError());\n\
              \x20  cublasDestroy(handle);\n\
             }\n"
                .to_string(),
        KernelVariant::Wmma => emit_matmul_wmma_body(k, _mm, &wmma_step1_config()),
        KernelVariant::KiviFlash => {
            unreachable!("KiviFlash variant does not support matmul")
        }
        KernelVariant::PagedFlash => {
            unreachable!("PagedFlash variant does not support matmul")
        }
    }
}

pub(crate) fn emit_matmul(k: &Kernel, mm: &Matmul, variant: KernelVariant) -> String {
    // WMMA has its own complete emitter; delegate immediately.
    if variant == KernelVariant::Wmma {
        return emit_matmul_wmma(k, mm, &wmma_step1_config());
    }
    let name = &k.name;
    let (note, includes, kernel_def, compute) = match variant {
        KernelVariant::Naive => (
            "Naive one-thread-per-output-element kernel with fp32 accumulation.",
            "",
            matmul_device_kernel(name),
            "    dim3 block(16, 16);\n\
             \x20   dim3 grid((K + 15) / 16, (M + 15) / 16);\n\
             \x20   {name}_kernel<<<grid, block>>>(dA, dB, dC);\n\
             \x20   CUDA_CHECK(cudaGetLastError());"
                .replace("{name}", name),
        ),
        KernelVariant::Cublas => (
            "cuBLAS-delegated matmul (cublasGemmEx; row-major C = A·B via the operand-swap trick).",
            "#include <cublas_v2.h>\n",
            String::new(),
            // Row-major C[M,K]=A[M,N]·B[N,K] ⇒ col-major Cᵀ[K,M]=Bᵀ·Aᵀ: GemmEx(N,N,K,M,N, B(ld=K), A(ld=N), C(ld=K)).
            "    cublasHandle_t handle;\n\
             \x20   cublasCreate(&handle);\n\
             \x20   const float alpha = 1.0f, beta = 0.0f;\n\
             \x20   cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,\n\
             \x20                &alpha, dB, CUDA_R_16F, K, dA, CUDA_R_16F, N,\n\
             \x20                &beta, dC, CUDA_R_32F, K, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);\n\
             \x20   CUDA_CHECK(cudaGetLastError());\n\
             \x20   cublasDestroy(handle);"
                .to_string(),
        ),
        KernelVariant::Wmma => unreachable!("handled above"),
        KernelVariant::KiviFlash => {
            unreachable!("KiviFlash variant does not support matmul")
        }
        KernelVariant::PagedFlash => {
            unreachable!("PagedFlash variant does not support matmul")
        }
    };
    format!(
        r#"// Generated by jacq — kernel `{name}` (matmul: C[{m},{kk}] = sum_n A[{m},{nn}] * B[{nn},{kk}]).
// {note} Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
{includes}#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int M = {m_size};  // {m}
static constexpr int N = {n_size};  // {nn}: contraction
static constexpr int K = {k_size};  // {kk}

{kernel_def}static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 4) {{
        fprintf(stderr, "usage: %s <A.bin [M*N fp16]> <B.bin [N*K fp16]> <C.bin [M*K fp32 out]>\n", argv[0]);
        return 1;
    }}
    __half* dA = load_half(argv[1], (size_t)M * N);
    __half* dB = load_half(argv[2], (size_t)N * K);
    float* dC;
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * K * sizeof(float)));

{compute}

    float* hC = (float*)malloc((size_t)M * K * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hC, dC, (size_t)M * K * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[3], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }}
    fwrite(hC, sizeof(float), (size_t)M * K, out);
    fclose(out);
    printf("C[0:8] =");
    for (int i = 0; i < 8 && i < M * K; ++i) printf(" %.4f", hC[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        note = note,
        includes = includes,
        kernel_def = kernel_def,
        compute = compute,
        m = mm.m_dim,
        nn = mm.n_dim,
        kk = mm.k_dim,
        m_size = mm.m,
        n_size = mm.n,
        k_size = mm.k,
    )
}

// ---- Softmax --------------------------------------------------------------------

/// The pieces of a recognized row-wise softmax kernel: `y[r, :] = softmax(x[r, :])`.
pub(crate) struct Softmax {
    pub(crate) rows_dim: String,
    pub(crate) cols_dim: String,
    pub(crate) rows: i64,
    pub(crate) cols: i64,
}

/// Recognize a row-wise softmax: a 2-D output `[R, C]` (`-> global`), a 2-D input `[R, C]`, and a
/// `DST = softmax(SRC, dim: C)` statement reducing over the column (last) dimension.
pub(crate) fn match_softmax(k: &Kernel) -> Option<Softmax> {
    let out = k.tensors.iter().find(|t| t.output && t.dims.len() == 2)?;
    let (rows_dim, cols_dim) = (out.dims[0].clone(), out.dims[1].clone());
    let input = k.tensors.iter().find(|t| {
        !t.output && t.dims == [rows_dim.clone(), cols_dim.clone()] && t.name != out.name
    })?;
    let _ = input;
    // A softmax statement over the column dim must be present.
    let over_cols = k
        .phases
        .iter()
        .flat_map(|p| &p.body)
        .any(|s| matches!(s, KStmt::Softmax { dim, .. } if *dim == cols_dim));
    if !over_cols {
        return None;
    }
    Some(Softmax {
        rows: param(k, &rows_dim)?,
        cols: param(k, &cols_dim)?,
        rows_dim,
        cols_dim,
    })
}

/// Shared `__global__` device-kernel source for softmax.
/// Returns the kernel function ending with `}\n\n`; no preamble, no host code.
pub(crate) fn softmax_device_kernel(name: &str) -> String {
    format!(
        "// Block-wide reduction over `red` (one slot per warp); `combine` is fmaxf or addition.\n\
         __global__ void {name}_kernel(const __half* x, float* y) {{\n\
         \x20   int row = blockIdx.x;\n\
         \x20   if (row >= R) return;\n\
         \x20   const __half* xr = x + (size_t)row * C;\n\
         \x20   float* yr = y + (size_t)row * C;\n\
         \x20   __shared__ float red[32];\n\
         \x20   int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarps = (blockDim.x + 31) / 32;\n\
         \n\
         \x20   // 1. row max (stable softmax).\n\
         \x20   float v = -1e30f;\n\
         \x20   for (int i = threadIdx.x; i < C; i += blockDim.x) v = fmaxf(v, __half2float(xr[i]));\n\
         \x20   for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));\n\
         \x20   if (lane == 0) red[wid] = v;\n\
         \x20   __syncthreads();\n\
         \x20   if (threadIdx.x == 0) {{ float m = -1e30f; for (int w = 0; w < nwarps; ++w) m = fmaxf(m, red[w]); red[0] = m; }}\n\
         \x20   __syncthreads();\n\
         \x20   float rmax = red[0];\n\
         \n\
         \x20   // 2. exp into yr, accumulate the row sum.\n\
         \x20   float ssum = 0.0f;\n\
         \x20   for (int i = threadIdx.x; i < C; i += blockDim.x) {{ float e = expf(__half2float(xr[i]) - rmax); yr[i] = e; ssum += e; }}\n\
         \x20   for (int o = 16; o > 0; o >>= 1) ssum += __shfl_xor_sync(0xffffffff, ssum, o);\n\
         \x20   if (lane == 0) red[wid] = ssum;\n\
         \x20   __syncthreads();\n\
         \x20   if (threadIdx.x == 0) {{ float t = 0.0f; for (int w = 0; w < nwarps; ++w) t += red[w]; red[0] = t; }}\n\
         \x20   __syncthreads();\n\
         \x20   float total = red[0];\n\
         \n\
         \x20   // 3. normalize.\n\
         \x20   for (int i = threadIdx.x; i < C; i += blockDim.x) yr[i] = yr[i] / total;\n\
         }}\n\n"
    )
}

/// Emit just the device kernel and a stream-aware `jac_launch` wrapper for softmax.
/// Parameter order for `jac_launch`: input x[R,C], then output y[R,C], then stream.
pub(crate) fn emit_softmax_body(k: &Kernel, _s: &Softmax) -> String {
    let name = &k.name;
    format!(
        "{kernel}\
         // jac_launch: input x[R,C], then output y[R,C], then stream.\n\
         extern \"C\" void jac_launch(const __half* x, float* y, cudaStream_t stream) {{\n\
         \x20   {name}_kernel<<<R, 256, 0, stream>>>(x, y);\n\
         \x20   CUDA_CHECK(cudaGetLastError());\n\
         }}\n",
        kernel = softmax_device_kernel(name),
    )
}

pub(crate) fn emit_softmax(k: &Kernel, s: &Softmax) -> String {
    let name = &k.name;
    format!(
        r#"// Generated by jacq — kernel `{name}` (row-wise softmax over {cols}: y[{rows},{cols}]).
// One block per row; numerically stable (subtract row max). Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int R = {rows_size};  // {rows}: rows
static constexpr int C = {cols_size};  // {cols}: softmax dimension

// Block-wide reduction over `red` (one slot per warp); `combine` is fmaxf or addition.
__global__ void {name}_kernel(const __half* x, float* y) {{
    int row = blockIdx.x;
    if (row >= R) return;
    const __half* xr = x + (size_t)row * C;
    float* yr = y + (size_t)row * C;
    __shared__ float red[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarps = (blockDim.x + 31) / 32;

    // 1. row max (stable softmax).
    float v = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) v = fmaxf(v, __half2float(xr[i]));
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (threadIdx.x == 0) {{ float m = -1e30f; for (int w = 0; w < nwarps; ++w) m = fmaxf(m, red[w]); red[0] = m; }}
    __syncthreads();
    float rmax = red[0];

    // 2. exp into yr, accumulate the row sum.
    float ssum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {{ float e = expf(__half2float(xr[i]) - rmax); yr[i] = e; ssum += e; }}
    for (int o = 16; o > 0; o >>= 1) ssum += __shfl_xor_sync(0xffffffff, ssum, o);
    if (lane == 0) red[wid] = ssum;
    __syncthreads();
    if (threadIdx.x == 0) {{ float t = 0.0f; for (int w = 0; w < nwarps; ++w) t += red[w]; red[0] = t; }}
    __syncthreads();
    float total = red[0];

    // 3. normalize.
    for (int i = threadIdx.x; i < C; i += blockDim.x) yr[i] = yr[i] / total;
}}

static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 3) {{
        fprintf(stderr, "usage: %s <x.bin [R*C fp16]> <y.bin [R*C fp32 out]>\n", argv[0]);
        return 1;
    }}
    __half* dx = load_half(argv[1], (size_t)R * C);
    float* dy;
    CUDA_CHECK(cudaMalloc(&dy, (size_t)R * C * sizeof(float)));
    {name}_kernel<<<R, 256>>>(dx, dy);
    CUDA_CHECK(cudaGetLastError());
    float* hy = (float*)malloc((size_t)R * C * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hy, dy, (size_t)R * C * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[2], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }}
    fwrite(hy, sizeof(float), (size_t)R * C, out);
    fclose(out);
    printf("y[0:8] =");
    for (int i = 0; i < 8 && i < R * C; ++i) printf(" %.4f", hy[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        rows = s.rows_dim,
        cols = s.cols_dim,
        rows_size = s.rows,
        cols_size = s.cols,
    )
}

// ---- Attention ------------------------------------------------------------------

/// The pieces of a recognized decode-attention kernel.
///
/// Decode shape (batch = 1, one query per head): `O[NH,HD] = softmax(Q[NH,HD] · K[NKV,L,HD]ᵀ / √HD) · V[NKV,L,HD]`,
/// with GQA mapping `kv_head = head / heads_per_group` (`heads_per_group = NH / NKV`). This BASE
/// instance is EXACT attention (reads all `L`); windowing/compaction are additive variants.
pub(crate) struct Attention {
    pub(crate) nh_dim: String,
    pub(crate) nkv_dim: String,
    pub(crate) l_dim: String,
    pub(crate) hd_dim: String,
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) l: i64,
    pub(crate) hd: i64,
    /// Sink + window policy (kept = first SINK + last WINDOW tokens). `Some` for windowed
    /// attention, `None` for the exact (read-all-L) base variant. Both present, or neither.
    pub(crate) sink: Option<i64>,
    pub(crate) window: Option<i64>,
}

/// Recognize decode attention: a 2-D output `O[NH,HD]` (`-> global`), a 2-D query `Q[NH,HD]`
/// sharing O's shape, two 3-D KV tensors `K,V[NKV,L,HD]` sharing HD with O, and a softmax that
/// reduces over the context dim `L`. NH must be a multiple of NKV (GQA grouping).
pub(crate) fn match_attention(k: &Kernel) -> Option<Attention> {
    // Output O[NH, HD]: the 2-D `-> global` tensor.
    let out = k.tensors.iter().find(|t| t.output && t.dims.len() == 2)?;
    let (nh_dim, hd_dim) = (out.dims[0].clone(), out.dims[1].clone());
    // Query Q[NH, HD]: a 2-D input sharing O's shape.
    let q = k
        .tensors
        .iter()
        .find(|t| !t.output && t.dims == [nh_dim.clone(), hd_dim.clone()] && t.name != out.name)?;
    // K[NKV, L, HD]: a 3-D input whose last (head) dim is HD.
    let kt = k
        .tensors
        .iter()
        .find(|t| !t.output && t.dims.len() == 3 && t.dims[2] == hd_dim && t.name != q.name)?;
    let (nkv_dim, l_dim) = (kt.dims[0].clone(), kt.dims[1].clone());
    // V[NKV, L, HD]: a second 3-D input sharing K's shape.
    k.tensors
        .iter()
        .find(|t| !t.output && t.dims == kt.dims && t.name != kt.name)?;
    // The scores normalization: a softmax reducing over the context dim L must be present.
    let over_l = k
        .phases
        .iter()
        .flat_map(|p| &p.body)
        .any(|s| matches!(s, KStmt::Softmax { dim, .. } if *dim == l_dim));
    if !over_l {
        return None;
    }
    let (nh, nkv, l, hd) = (
        param(k, &nh_dim)?,
        param(k, &nkv_dim)?,
        param(k, &l_dim)?,
        param(k, &hd_dim)?,
    );
    // GQA grouping: NH query heads must partition evenly over NKV kv-heads.
    if nkv <= 0 || nh % nkv != 0 {
        return None;
    }
    // Optional sink+window policy: windowed iff BOTH SINK and WINDOW params are present. A partial
    // policy (only one) is malformed — reject rather than silently emit exact attention.
    let (sink, window) = (param(k, "SINK"), param(k, "WINDOW"));
    if sink.is_some() != window.is_some() {
        return None;
    }
    Some(Attention {
        nh_dim,
        nkv_dim,
        l_dim,
        hd_dim,
        nh,
        nkv,
        l,
        hd,
        sink,
        window,
    })
}

/// Shared `__global__` device kernels for decode attention (EXACT, reads all L) — the three
/// stages `score -> softmax -> value`, GQA-mapped (`kv_head = head / HEADS_PER_GROUP`), one block
/// per head. Structurally this is the hand-written `masked.cu` reference with the window mask
/// removed (i.e. the no-op-mask point), so it is the correctness anchor. `__KNAME__` is replaced
/// by the kernel name; `NH`/`NKV`/`L`/`HD` are preprocessor macros provided by the harness (or
/// `constexpr` in the standalone path). Shared by `emit_attention_body` and `emit_attention`.
pub(crate) fn attention_device_kernels(name: &str, policy: Option<(i64, i64)>) -> String {
    const KERNELS: &str = r#"#define HEADS_PER_GROUP (NH / NKV)
__KEPT_HELPER__
// scores[h*L + s] = (Q[h] . K[kv_h,s]) / sqrt(HD). __SCORE_COMMENT__
__global__ void __KNAME___score(const __half* q, const __half* k, float* scores) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const __half* qh = q + (size_t)h * HD;
    const __half* kbase = k + (size_t)kv_h * L * HD;
    float scale = rsqrtf((float)HD);
    for (int s = threadIdx.x; s < L; s += blockDim.x) {
        const __half* ks = kbase + (size_t)s * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
__SCORE_STORE__
    }
}

// In-place numerically-stable softmax over scores[h, 0..L). One block per head.
__global__ void __KNAME___softmax(float* scores) {
    int h = blockIdx.x; float* row = scores + (size_t)h * L; int tid = threadIdx.x;
    __shared__ float red[32];
    float m = -1e30f;
    for (int s = tid; s < L; s += blockDim.x) m = fmaxf(m, row[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) { int nw = (blockDim.x + 31) / 32; float mm = red[0]; for (int i = 1; i < nw; ++i) mm = fmaxf(mm, red[i]); red[0] = mm; }
    __syncthreads();
    float mx = red[0], sum = 0.0f;
    for (int s = tid; s < L; s += blockDim.x) { float e = expf(row[s] - mx); row[s] = e; sum += e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) { int nw = (blockDim.x + 31) / 32; float ss = 0.0f; for (int i = 0; i < nw; ++i) ss += red[i]; red[0] = ss; }
    __syncthreads();
    float inv = 1.0f / red[0];
    for (int s = tid; s < L; s += blockDim.x) row[s] *= inv;
}

// O[h,d] = sum_s scores[h,s] * V[kv_h,s,d]. Reads all V. fp32 output.
__global__ void __KNAME___value(float* out, const float* scores, const __half* v) {
    int h = blockIdx.x, kv_h = h / HEADS_PER_GROUP;
    const float* srow = scores + (size_t)h * L;
    const __half* vbase = v + (size_t)kv_h * L * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < L; ++s)
            acc += srow[s] * __half2float(vbase[(size_t)s * HD + d]);
        out[(size_t)h * HD + d] = acc;
    }
}

"#;
    // The window policy touches ONLY the score kernel: keep the first SINK + last WINDOW keys, mask
    // the dropped middle to -INF (softmax then zeroes them, value reads contribute 0). Exact and
    // windowed share the softmax/value kernels verbatim. Exact output is byte-identical to before.
    let windowed = policy.is_some();
    // The body SELF-defines SINK/WINDOW: the experiment harness only -D-defines tensor-DIMENSION
    // params (NH/HD/NKV/L), not policy scalars, so the generated kernel must carry its own defaults
    // (like the hand-written masked.cu). `#ifndef`-guarded so a sweep can still override via -D.
    let kept_helper = match policy {
        Some((sink, window)) => format!(
            "\n#ifndef SINK\n#define SINK {sink}\n#endif\n#ifndef WINDOW\n#define WINDOW {window}\n#endif\n// kept(s): keep the first SINK + the last WINDOW tokens of the L-long cache.\n__device__ __forceinline__ bool attn_kept(int s) {{ return (s < SINK) || (s >= L - WINDOW); }}"
        ),
        None => String::new(),
    };
    let score_comment = if windowed {
        "WINDOWED: keep first SINK + last WINDOW, drop the middle to -INF."
    } else {
        "EXACT attention: reads all L keys."
    };
    let score_store = if windowed {
        "        float sv = dot * scale;\n        if (!attn_kept(s)) sv = -INFINITY;   // drop the middle (sink+window policy)\n        scores[(size_t)h * L + s] = sv;"
    } else {
        "        scores[(size_t)h * L + s] = dot * scale;"
    };
    KERNELS
        .replace("__KEPT_HELPER__", &kept_helper)
        .replace("__SCORE_COMMENT__", score_comment)
        .replace("__SCORE_STORE__", score_store)
        .replace("__KNAME__", name)
}

/// Emit just the device kernels + a stream-aware `jac_launch` for decode attention (the
/// preamble-free body used by the experiment harness via `emit_kernel_body`).
///
/// `jac_launch` ABI: non-output inputs in declaration order (Q, K, V), then output O, then stream.
/// The `scores[NH,L]` scratch is self-managed (a `static` lazy `cudaMalloc` on the first warmup
/// call): a Jac subject's body is `#include`d into the harness TU, which already weak-defines
/// `jac_workspace_bytes` — nvcc rejects a second definition in the same TU.
pub(crate) fn emit_attention_body(k: &Kernel, a: &Attention) -> String {
    let name = &k.name;
    let kernels = attention_device_kernels(name, a.sink.zip(a.window));
    let launch = r#"// jac_launch: non-output inputs in declaration order (Q[NH,HD], K[NKV,L,HD], V[NKV,L,HD]), then O[NH,HD], then stream.
extern "C" void jac_launch(const __half* Q, const __half* K, const __half* V, float* O, cudaStream_t stream) {
    static float* scores = nullptr;
    if (!scores) cudaMalloc((void**)&scores, (size_t)NH * L * sizeof(float));  // self-managed scratch (warmup-allocated)
    __KNAME___score <<<NH, 128, 0, stream>>>(Q, K, scores);
    __KNAME___softmax <<<NH, 128, 0, stream>>>(scores);
    __KNAME___value <<<NH, 64, 0, stream>>>(O, scores, V);
    CUDA_CHECK(cudaGetLastError());
}
"#
    .replace("__KNAME__", name);
    format!("{kernels}{launch}")
}

/// Emit the standalone `.cu` (device kernels + `main()` harness) for decode attention — the
/// `jacq --emit-cuda` path. Reads Q/K/V from fp16 files, writes O as fp32. The `main()` allocates
/// the `scores[NH,L]` scratch directly (it owns the whole TU, so no weak-default collision).
pub(crate) fn emit_attention(k: &Kernel, a: &Attention) -> String {
    let name = &k.name;
    let kernels = attention_device_kernels(name, a.sink.zip(a.window));
    format!(
        r#"// Generated by jacq — kernel `{name}` (decode attention: O[{nh_dim},{hd_dim}] = softmax(Q.K^T / sqrt(HD)) . V).
// EXACT attention (reads all {l_dim}); GQA kv_head = head / (NH/NKV); one block per head. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};   // {nh_dim}: query heads
static constexpr int HD = {hd};   // {hd_dim}: head dim
static constexpr int NKV = {nkv};  // {nkv_dim}: kv heads (GQA)
static constexpr int L = {l};    // {l_dim}: context length

{kernels}static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 5) {{
        fprintf(stderr, "usage: %s <Q.bin [NH*HD fp16]> <K.bin [NKV*L*HD fp16]> <V.bin [NKV*L*HD fp16]> <O.bin [NH*HD fp32 out]>\n", argv[0]);
        return 1;
    }}
    __half* dQ = load_half(argv[1], (size_t)NH * HD);
    __half* dK = load_half(argv[2], (size_t)NKV * L * HD);
    __half* dV = load_half(argv[3], (size_t)NKV * L * HD);
    float* dScores;
    CUDA_CHECK(cudaMalloc(&dScores, (size_t)NH * L * sizeof(float)));
    float* dO;
    CUDA_CHECK(cudaMalloc(&dO, (size_t)NH * HD * sizeof(float)));

    {name}_score <<<NH, 128>>>(dQ, dK, dScores);
    {name}_softmax <<<NH, 128>>>(dScores);
    {name}_value <<<NH, 64>>>(dO, dScores, dV);
    CUDA_CHECK(cudaGetLastError());

    float* hO = (float*)malloc((size_t)NH * HD * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hO, dO, (size_t)NH * HD * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[4], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }}
    fwrite(hO, sizeof(float), (size_t)NH * HD, out);
    fclose(out);
    printf("O[0:8] =");
    for (int i = 0; i < 8 && i < NH * HD; ++i) printf(" %.4f", hO[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        l = a.l,
        nh_dim = a.nh_dim,
        hd_dim = a.hd_dim,
        nkv_dim = a.nkv_dim,
        l_dim = a.l_dim,
        kernels = kernels,
    )
}

// ---- KIVI int4 decode attention -------------------------------------------------

/// Recognized first-lane KIVI decode-attention shape.
///
/// This is the DSL-side ABI for the faithful KIVI4 kernel family:
/// keys are grouped over tokens per channel, values are grouped over channels per token, both use
/// asymmetric int4 codes plus fp16 scale/zero, and the recent residual window remains fp16.  The
/// emitter below fuses score/softmax/value into one launch and never materializes `scores`, which
/// is the important envelope even though it is not yet the split-K/GQA-amortized
/// schedule.
pub(crate) struct KiviAttention {
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) lq: i64,
    pub(crate) residual: i64,
    pub(crate) group: i64,
    pub(crate) hd_packed: i64,
    pub(crate) lq_groups: i64,
    pub(crate) hd_groups: i64,
}

/// Recognized source-vLLM paged KIVI decode ABI.
///
/// This is the next serving-lane shape after the contiguous KIVI4 decode ABI: old quantized
/// KV is stored in physical vLLM pages and addressed through a block table.  It is still a first
/// single-sequence decode lane, not a full batched/split-K backend.
pub(crate) struct KiviPagedAttention {
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) lq: i64,
    pub(crate) residual: i64,
    pub(crate) group: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    pub(crate) hd_packed: i64,
    pub(crate) hd_groups: i64,
}

/// Recognized source-vLLM sparse-retained paged KIVI decode ABI.
///
/// Same physical page tensors as [`KiviPagedAttention`], but the block table is a compact list of
/// retained physical page ids after page-aware eviction. The kernel loops only over
/// `NUM_RETAINED_PAGES * BLOCK_SIZE` old tokens and keeps residual-window length tied to the
/// original dense `SEQ_LEN`, so evicted old pages do not silently re-enter via residual scratch.
pub(crate) struct KiviPagedSparseRetainedAttention {
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) lq: i64,
    pub(crate) residual: i64,
    pub(crate) group: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    pub(crate) hd_packed: i64,
    pub(crate) hd_groups: i64,
}

/// Recognized source-vLLM BATCHED paged KIVI decode ABI (batched prefill-loop fix).
///
/// Batched sibling of [`KiviPagedAttention`]: same per-query online-softmax math, but Q/O gain a
/// leading MAX_QUERIES dimension and the single-token spec's scalar runtime SEQ_LEN becomes a
/// SEQ_LENS[MAX_QUERIES] tensor, so one launch can cover a whole prefill sub-chunk of new tokens
/// instead of one launch per token. Per-query output is expected to be bit-identical to the
/// single-token kernel for the same (head, seq_len) pair; that parity is an expectation of this
/// repository, not a completed gate, and live request-loop ordering behavior is likewise
/// unproven here -- both require the GPU validation gates described in docs/GPU_VALIDATION.md.
pub(crate) struct KiviPagedBatchedAttention {
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) lq: i64,
    pub(crate) residual: i64,
    pub(crate) group: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    pub(crate) hd_packed: i64,
    pub(crate) hd_groups: i64,
    pub(crate) max_queries: i64,
}

/// Recognized source-vLLM paged KIVI cache-writer ABI (DSL-ize the write side).
///
/// Write-side sibling of [`KiviPagedAttention`]: no Q/O attention tensors, since this kernel
/// appends one new token's K/V into residual scratch and conditionally quantizes an aged-out
/// 32-token group into a BLOCK_TABLE-addressed physical page, rather than computing attention.
pub(crate) struct KiviCacheWrite {
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) residual: i64,
    pub(crate) residual_scratch: i64,
    pub(crate) group: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    pub(crate) hd_packed: i64,
    pub(crate) hd_groups: i64,
}

/// Recognized source-vLLM BATCHED paged KIVI cache-writer ABI.
///
/// Write-side sibling of [`KiviCacheWrite`]: NEW_K/NEW_V carry a leading MAX_NEW token
/// dimension so one prefill sub-chunk appends multiple tokens before an optional one-page age-out.
pub(crate) struct KiviCacheWriteBatched {
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) residual: i64,
    pub(crate) residual_scratch: i64,
    pub(crate) group: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    pub(crate) hd_packed: i64,
    pub(crate) hd_groups: i64,
    pub(crate) max_new: i64,
}

fn tensor<'a>(k: &'a Kernel, name: &str) -> Option<&'a jac_syntax::ast::TensorDecl> {
    k.tensors.iter().find(|t| t.name == name)
}

/// Recognize the KIVI int4 decode-attention ABI by explicit tensor names.
///
/// The current kernel DSL has no first-class `dequant_kivi4(...)` statement, so the first honest
/// lowering seam is an ABI-level pattern: if a spec names the KIVI code/scale/zero/residual
/// tensors and declares the KIVI params, codegen emits the fused int4 decode body.
pub(crate) fn match_kivi_attention(k: &Kernel) -> Option<KiviAttention> {
    let q = tensor(k, "Q")?;
    let kc = tensor(k, "K_CODE")?;
    let ks = tensor(k, "K_SCALE")?;
    let kz = tensor(k, "K_ZERO")?;
    let vc = tensor(k, "V_CODE")?;
    let vs = tensor(k, "V_SCALE")?;
    let vz = tensor(k, "V_ZERO")?;
    let kr = tensor(k, "K_RESIDUAL")?;
    let vr = tensor(k, "V_RESIDUAL")?;
    let o = k.tensors.iter().find(|t| t.output && t.name == "O")?;

    let (nh, hd, nkv, lq, residual, group, hd_packed, lq_groups, hd_groups) = (
        param(k, "NH")?,
        param(k, "HD")?,
        param(k, "NKV")?,
        param(k, "LQ")?,
        param(k, "RESIDUAL")?,
        param(k, "KIVI_GROUP")?,
        param(k, "HD_PACKED")?,
        param(k, "LQ_GROUPS")?,
        param(k, "HD_GROUPS")?,
    );
    if nh <= 0
        || nkv <= 0
        || hd <= 0
        || lq < 0
        || residual < 0
        || group <= 0
        || hd != 128
        || nh % nkv != 0
        || hd % group != 0
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || lq_groups != (lq + group - 1) / group
    {
        return None;
    }
    if q.dims != ["NH".to_string(), "HD".to_string()]
        || o.dims != ["NH".to_string(), "HD".to_string()]
        || o.dtype != "fp32"
        || kc.dims != ["NKV".to_string(), "LQ".to_string(), "HD_PACKED".to_string()]
        || vc.dims != kc.dims
        || ks.dims != ["NKV".to_string(), "LQ_GROUPS".to_string(), "HD".to_string()]
        || kz.dims != ks.dims
        || vs.dims != ["NKV".to_string(), "LQ".to_string(), "HD_GROUPS".to_string()]
        || vz.dims != vs.dims
        || kr.dims != ["NKV".to_string(), "RESIDUAL".to_string(), "HD".to_string()]
        || vr.dims != kr.dims
        || q.dtype != "fp16"
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
        || kr.dtype != "fp16"
        || vr.dtype != "fp16"
    {
        return None;
    }

    Some(KiviAttention {
        nh,
        nkv,
        hd,
        lq,
        residual,
        group,
        hd_packed,
        lq_groups,
        hd_groups,
    })
}

/// Recognize the source-vLLM paged KIVI int4 decode ABI.
///
/// The page ABI deliberately uses `BLOCK_SIZE == KIVI_GROUP == 32` so key token-groups never
/// straddle pages.  `BLOCK_TABLE[logical_page] -> physical_page` mirrors vLLM's paged cache
/// boundary while the residual window remains separate fp16 storage for the first lane.
pub(crate) fn match_kivi_paged_attention(k: &Kernel) -> Option<KiviPagedAttention> {
    let q = tensor(k, "Q")?;
    let bt = tensor(k, "BLOCK_TABLE")?;
    let kc = tensor(k, "K_CODE_PAGES")?;
    let ks = tensor(k, "K_SCALE_PAGES")?;
    let kz = tensor(k, "K_ZERO_PAGES")?;
    let vc = tensor(k, "V_CODE_PAGES")?;
    let vs = tensor(k, "V_SCALE_PAGES")?;
    let vz = tensor(k, "V_ZERO_PAGES")?;
    let o = k.tensors.iter().find(|t| t.output && t.name == "O")?;

    let (
        nh,
        hd,
        nkv,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    ) = (
        param(k, "NH")?,
        param(k, "HD")?,
        param(k, "NKV")?,
        param(k, "LQ")?,
        param(k, "RESIDUAL")?,
        param(k, "KIVI_GROUP")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
        param(k, "HD_PACKED")?,
        param(k, "HD_GROUPS")?,
    );
    if nh <= 0
        || nkv <= 0
        || hd <= 0
        || lq < 0
        || residual < 0
        || group <= 0
        || block_size != group
        || block_size != 32
        || hd != 128
        || nh % nkv != 0
        || hd % group != 0
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || max_blocks <= 0
        || num_blocks <= 0
        || lq > max_blocks * block_size
    {
        return None;
    }
    if q.dims != ["NH".to_string(), "HD".to_string()]
        || o.dims != ["NH".to_string(), "HD".to_string()]
        || o.dtype != "fp32"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kc.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_PACKED".to_string(),
            ]
        || vc.dims != kc.dims
        || ks.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "HD".to_string(),
            ]
        || kz.dims != ks.dims
        || vs.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_GROUPS".to_string(),
            ]
        || vz.dims != vs.dims
        || q.dtype != "fp16"
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
    {
        return None;
    }

    Some(KiviPagedAttention {
        nh,
        nkv,
        hd,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    })
}

/// Recognize the source-vLLM sparse-retained paged KIVI int4 decode ABI.
///
/// This is intentionally disjoint from [`match_kivi_paged_attention`] by tensor name:
/// `RETAINED_BLOCK_TABLE` is a compact table of retained physical pages, not the dense vLLM
/// `BLOCK_TABLE[logical_page]` covering every old page. The runtime scalar
/// `NUM_RETAINED_PAGES` is introduced by the emitted `jac_launch` ABI and bounded by this
/// `MAX_BLOCKS` shape.
pub(crate) fn match_kivi_paged_sparse_retained_attention(
    k: &Kernel,
) -> Option<KiviPagedSparseRetainedAttention> {
    let q = tensor(k, "Q")?;
    let bt = tensor(k, "RETAINED_BLOCK_TABLE")?;
    let kc = tensor(k, "K_CODE_PAGES")?;
    let ks = tensor(k, "K_SCALE_PAGES")?;
    let kz = tensor(k, "K_ZERO_PAGES")?;
    let vc = tensor(k, "V_CODE_PAGES")?;
    let vs = tensor(k, "V_SCALE_PAGES")?;
    let vz = tensor(k, "V_ZERO_PAGES")?;
    let o = k.tensors.iter().find(|t| t.output && t.name == "O")?;

    let (
        nh,
        hd,
        nkv,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    ) = (
        param(k, "NH")?,
        param(k, "HD")?,
        param(k, "NKV")?,
        param(k, "LQ")?,
        param(k, "RESIDUAL")?,
        param(k, "KIVI_GROUP")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
        param(k, "HD_PACKED")?,
        param(k, "HD_GROUPS")?,
    );
    if nh <= 0
        || nkv <= 0
        || hd <= 0
        || lq < 0
        || residual < 0
        || group <= 0
        || block_size != group
        || block_size != 32
        || hd != 128
        || nh % nkv != 0
        || hd % group != 0
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || max_blocks <= 0
        || num_blocks <= 0
        || lq > max_blocks * block_size
    {
        return None;
    }
    if q.dims != ["NH".to_string(), "HD".to_string()]
        || o.dims != ["NH".to_string(), "HD".to_string()]
        || o.dtype != "fp32"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kc.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_PACKED".to_string(),
            ]
        || vc.dims != kc.dims
        || ks.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "HD".to_string(),
            ]
        || kz.dims != ks.dims
        || vs.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_GROUPS".to_string(),
            ]
        || vz.dims != vs.dims
        || q.dtype != "fp16"
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
    {
        return None;
    }

    Some(KiviPagedSparseRetainedAttention {
        nh,
        nkv,
        hd,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    })
}

/// Recognize the source-vLLM BATCHED paged KIVI int4 decode ABI (batched prefill-loop fix).
///
/// Distinguished from [`match_kivi_paged_attention`] by Q/O carrying a leading MAX_QUERIES
/// dimension and a SEQ_LENS[MAX_QUERIES] tensor in place of the single-token spec's scalar
/// runtime SEQ_LEN -- a different spec (decode_attention_kivi_paged_batched.jac) than the
/// single-token one, so this matcher cannot accidentally fire on the existing, unchanged spec.
pub(crate) fn match_kivi_paged_batched_attention(k: &Kernel) -> Option<KiviPagedBatchedAttention> {
    let q = tensor(k, "Q")?;
    let seq_lens = tensor(k, "SEQ_LENS")?;
    let bt = tensor(k, "BLOCK_TABLE")?;
    let kc = tensor(k, "K_CODE_PAGES")?;
    let ks = tensor(k, "K_SCALE_PAGES")?;
    let kz = tensor(k, "K_ZERO_PAGES")?;
    let vc = tensor(k, "V_CODE_PAGES")?;
    let vs = tensor(k, "V_SCALE_PAGES")?;
    let vz = tensor(k, "V_ZERO_PAGES")?;
    let o = k.tensors.iter().find(|t| t.output && t.name == "O")?;

    let (
        nh,
        hd,
        nkv,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
        max_queries,
    ) = (
        param(k, "NH")?,
        param(k, "HD")?,
        param(k, "NKV")?,
        param(k, "LQ")?,
        param(k, "RESIDUAL")?,
        param(k, "KIVI_GROUP")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
        param(k, "HD_PACKED")?,
        param(k, "HD_GROUPS")?,
        param(k, "MAX_QUERIES")?,
    );
    if nh <= 0
        || nkv <= 0
        || hd <= 0
        || lq < 0
        || residual < 0
        || group <= 0
        || block_size != group
        || block_size != 32
        || hd != 128
        || nh % nkv != 0
        || hd % group != 0
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || max_blocks <= 0
        || num_blocks <= 0
        || lq > max_blocks * block_size
        || max_queries <= 0
    {
        return None;
    }
    if q.dims
        != [
            "MAX_QUERIES".to_string(),
            "NH".to_string(),
            "HD".to_string(),
        ]
        || o.dims != q.dims
        || o.dtype != "fp32"
        || seq_lens.dims != ["MAX_QUERIES".to_string()]
        || seq_lens.dtype != "int32"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kc.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_PACKED".to_string(),
            ]
        || vc.dims != kc.dims
        || ks.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "HD".to_string(),
            ]
        || kz.dims != ks.dims
        || vs.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_GROUPS".to_string(),
            ]
        || vz.dims != vs.dims
        || q.dtype != "fp16"
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
    {
        return None;
    }

    Some(KiviPagedBatchedAttention {
        nh,
        nkv,
        hd,
        lq,
        residual,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
        max_queries,
    })
}

/// Recognize the source-vLLM BATCHED paged KIVI cache-writer ABI.
///
/// Distinguished from the single-token writer by NEW_K/NEW_V shape: `[MAX_NEW, NKV, HD]`, not
/// `[NKV, HD]`. This is the DSL sibling of source_patch.py's hand-CUDA
/// `kivi_cache_write_batched_launch` used by `JAC_KIVI_BATCHED_PREFILL=1`.
pub(crate) fn match_kivi_cache_write_batched(k: &Kernel) -> Option<KiviCacheWriteBatched> {
    let new_k = tensor(k, "NEW_K")?;
    let new_v = tensor(k, "NEW_V")?;
    let bt = tensor(k, "BLOCK_TABLE")?;
    let kc = tensor(k, "K_CODE_PAGES")?;
    let ks = tensor(k, "K_SCALE_PAGES")?;
    let kz = tensor(k, "K_ZERO_PAGES")?;
    let vc = tensor(k, "V_CODE_PAGES")?;
    let vs = tensor(k, "V_SCALE_PAGES")?;
    let vz = tensor(k, "V_ZERO_PAGES")?;
    let krs = tensor(k, "K_RESIDUAL_SCRATCH")?;
    let vrs = tensor(k, "V_RESIDUAL_SCRATCH")?;

    let (
        nkv,
        hd,
        residual,
        residual_scratch,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
        max_new,
    ) = (
        param(k, "NKV")?,
        param(k, "HD")?,
        param(k, "RESIDUAL")?,
        param(k, "RESIDUAL_SCRATCH")?,
        param(k, "KIVI_GROUP")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
        param(k, "HD_PACKED")?,
        param(k, "HD_GROUPS")?,
        param(k, "MAX_NEW")?,
    );
    if nkv <= 0
        || hd <= 0
        || residual < 0
        || residual_scratch != residual + block_size
        || group <= 0
        || block_size != group
        || block_size != 32
        || hd != 128
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || max_blocks <= 0
        || num_blocks <= 0
        || max_new <= 0
        || max_new > residual_scratch
    {
        return None;
    }
    if new_k.dims != ["MAX_NEW".to_string(), "NKV".to_string(), "HD".to_string()]
        || new_v.dims != new_k.dims
        || new_k.dtype != "fp16"
        || new_v.dtype != "fp16"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kc.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_PACKED".to_string(),
            ]
        || vc.dims != kc.dims
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "HD".to_string(),
            ]
        || kz.dims != ks.dims
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_GROUPS".to_string(),
            ]
        || vz.dims != vs.dims
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
        || krs.dims
            != [
                "NKV".to_string(),
                "RESIDUAL_SCRATCH".to_string(),
                "HD".to_string(),
            ]
        || vrs.dims != krs.dims
        || krs.dtype != "fp16"
        || vrs.dtype != "fp16"
        || !kc.output
        || !ks.output
        || !kz.output
        || !vc.output
        || !vs.output
        || !vz.output
        || !krs.output
        || !vrs.output
    {
        return None;
    }

    Some(KiviCacheWriteBatched {
        nkv,
        hd,
        residual,
        residual_scratch,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
        max_new,
    })
}

/// Recognize the source-vLLM paged KIVI cache-writer ABI (DSL writer).
///
/// Distinguished from the decode matchers by tensor identity, not just shape: this spec declares
/// `NEW_K`/`NEW_V` (the new token being written) and has no `Q`/`O` attention tensors at all, so it
/// can never accidentally match a decode spec or vice versa.
pub(crate) fn match_kivi_cache_write(k: &Kernel) -> Option<KiviCacheWrite> {
    let new_k = tensor(k, "NEW_K")?;
    let new_v = tensor(k, "NEW_V")?;
    let bt = tensor(k, "BLOCK_TABLE")?;
    let kc = tensor(k, "K_CODE_PAGES")?;
    let ks = tensor(k, "K_SCALE_PAGES")?;
    let kz = tensor(k, "K_ZERO_PAGES")?;
    let vc = tensor(k, "V_CODE_PAGES")?;
    let vs = tensor(k, "V_SCALE_PAGES")?;
    let vz = tensor(k, "V_ZERO_PAGES")?;
    let krs = tensor(k, "K_RESIDUAL_SCRATCH")?;
    let vrs = tensor(k, "V_RESIDUAL_SCRATCH")?;

    let (
        nkv,
        hd,
        residual,
        residual_scratch,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    ) = (
        param(k, "NKV")?,
        param(k, "HD")?,
        param(k, "RESIDUAL")?,
        param(k, "RESIDUAL_SCRATCH")?,
        param(k, "KIVI_GROUP")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
        param(k, "HD_PACKED")?,
        param(k, "HD_GROUPS")?,
    );
    if nkv <= 0
        || hd <= 0
        || residual < 0
        || residual_scratch != residual + block_size
        || group <= 0
        || block_size != group
        || block_size != 32
        || hd != 128
        || hd_packed != hd / 2
        || hd_groups != (hd + group - 1) / group
        || max_blocks <= 0
        || num_blocks <= 0
    {
        return None;
    }
    if new_k.dims != ["NKV".to_string(), "HD".to_string()]
        || new_v.dims != new_k.dims
        || new_k.dtype != "fp16"
        || new_v.dtype != "fp16"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kc.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_PACKED".to_string(),
            ]
        || vc.dims != kc.dims
        || kc.dtype != "int8"
        || vc.dtype != "int8"
        || ks.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "HD".to_string(),
            ]
        || kz.dims != ks.dims
        || ks.dtype != "fp16"
        || kz.dtype != "fp16"
        || vs.dims
            != [
                "NUM_BLOCKS".to_string(),
                "NKV".to_string(),
                "BLOCK_SIZE".to_string(),
                "HD_GROUPS".to_string(),
            ]
        || vz.dims != vs.dims
        || vs.dtype != "fp16"
        || vz.dtype != "fp16"
        || krs.dims
            != [
                "NKV".to_string(),
                "RESIDUAL_SCRATCH".to_string(),
                "HD".to_string(),
            ]
        || vrs.dims != krs.dims
        || krs.dtype != "fp16"
        || vrs.dtype != "fp16"
        || !kc.output
        || !ks.output
        || !kz.output
        || !vc.output
        || !vs.output
        || !vz.output
        || !krs.output
        || !vrs.output
    {
        return None;
    }

    Some(KiviCacheWrite {
        nkv,
        hd,
        residual,
        residual_scratch,
        group,
        block_size,
        num_blocks,
        max_blocks,
        hd_packed,
        hd_groups,
    })
}

fn kivi_attention_device_kernel(name: &str) -> String {
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
static_assert(KIVI_GROUP == 32, "first KIVI DSL lane targets the KIVI group32 ABI");
static_assert(HD == 128, "first KIVI DSL lane is guarded to Qwen-family head_dim=128");
static_assert(HD_PACKED * 2 == HD, "HD_PACKED must store two int4 values per byte");
static_assert(HD_GROUPS == ((HD + KIVI_GROUP - 1) / KIVI_GROUP), "bad HD_GROUPS");
static_assert(LQ_GROUPS == ((LQ + KIVI_GROUP - 1) / KIVI_GROUP), "bad LQ_GROUPS");

__device__ __forceinline__ float kivi4_dequant(unsigned char byte, int d, float scale, float zero) {{
    int code = (d & 1) ? (byte >> 4) : (byte & 0x0f);
    return fmaf((float)code, scale, zero);
}}

__device__ __forceinline__ float block_sum(float x) {{
    __shared__ float red[32];
    int tid = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off);
    if ((tid & 31) == 0) red[tid >> 5] = x;
    __syncthreads();
    if (tid == 0) {{
        int nw = (blockDim.x + 31) / 32;
        float y = 0.0f;
        for (int i = 0; i < nw; ++i) y += red[i];
        red[0] = y;
    }}
    __syncthreads();
    return red[0];
}}

// Fused KIVI4 decode attention: score -> softmax -> value in one launch, no scores scratch.
// Quantized old cache: K_CODE/K_SCALE/K_ZERO + V_CODE/V_SCALE/V_ZERO. Recent RESIDUAL tokens stay fp16.
__global__ void {name}_kivi_decode(
    const __half* q,
    const unsigned char* k_code, const __half* k_scale, const __half* k_zero,
    const unsigned char* v_code, const __half* v_scale, const __half* v_zero,
    const __half* k_residual, const __half* v_residual,
    float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int tid = threadIdx.x;
    const __half* qh = q + (size_t)h * HD;
    float scale_attn = rsqrtf((float)HD);

    float row_max = -1e30f;
    for (int s = 0; s < LQ; ++s) {{
        float partial = 0.0f;
        int kg = s / KIVI_GROUP;
        for (int d = tid; d < HD; d += blockDim.x) {{
            unsigned char packed = k_code[((size_t)kv_h * LQ + s) * HD_PACKED + (d >> 1)];
            size_t pidx = ((size_t)kv_h * LQ_GROUPS + kg) * HD + d;
            float kd = kivi4_dequant(packed, d, __half2float(k_scale[pidx]), __half2float(k_zero[pidx]));
            partial += __half2float(qh[d]) * kd;
        }}
        float score = block_sum(partial) * scale_attn;
        if (tid == 0) row_max = fmaxf(row_max, score);
    }}
    for (int s = 0; s < RESIDUAL; ++s) {{
        float partial = 0.0f;
        const __half* ks = k_residual + ((size_t)kv_h * RESIDUAL + s) * HD;
        for (int d = tid; d < HD; d += blockDim.x)
            partial += __half2float(qh[d]) * __half2float(ks[d]);
        float score = block_sum(partial) * scale_attn;
        if (tid == 0) row_max = fmaxf(row_max, score);
    }}
    __shared__ float smem[2];  // [0] = row_max, [1] = softmax denom
    if (tid == 0) smem[0] = row_max;
    __syncthreads();
    row_max = smem[0];

    float denom = 0.0f;
    for (int s = 0; s < LQ; ++s) {{
        float partial = 0.0f;
        int kg = s / KIVI_GROUP;
        for (int d = tid; d < HD; d += blockDim.x) {{
            unsigned char packed = k_code[((size_t)kv_h * LQ + s) * HD_PACKED + (d >> 1)];
            size_t pidx = ((size_t)kv_h * LQ_GROUPS + kg) * HD + d;
            float kd = kivi4_dequant(packed, d, __half2float(k_scale[pidx]), __half2float(k_zero[pidx]));
            partial += __half2float(qh[d]) * kd;
        }}
        float score = block_sum(partial) * scale_attn;
        if (tid == 0) denom += expf(score - row_max);
    }}
    for (int s = 0; s < RESIDUAL; ++s) {{
        float partial = 0.0f;
        const __half* ks = k_residual + ((size_t)kv_h * RESIDUAL + s) * HD;
        for (int d = tid; d < HD; d += blockDim.x)
            partial += __half2float(qh[d]) * __half2float(ks[d]);
        float score = block_sum(partial) * scale_attn;
        if (tid == 0) denom += expf(score - row_max);
    }}
    if (tid == 0) smem[1] = fmaxf(denom, 1.0e-30f);
    __syncthreads();
    denom = smem[1];

    for (int d = tid; d < HD; d += blockDim.x) {{
        float acc = 0.0f;
        for (int s = 0; s < LQ; ++s) {{
            float partial = 0.0f;
            int kg = s / KIVI_GROUP;
            for (int dd = threadIdx.x; dd < HD; dd += blockDim.x) {{
                unsigned char packed_k = k_code[((size_t)kv_h * LQ + s) * HD_PACKED + (dd >> 1)];
                size_t kpidx = ((size_t)kv_h * LQ_GROUPS + kg) * HD + dd;
                float kd = kivi4_dequant(packed_k, dd, __half2float(k_scale[kpidx]), __half2float(k_zero[kpidx]));
                partial += __half2float(qh[dd]) * kd;
            }}
            float prob = expf(block_sum(partial) * scale_attn - row_max) / denom;
            unsigned char packed_v = v_code[((size_t)kv_h * LQ + s) * HD_PACKED + (d >> 1)];
            int vg = d / KIVI_GROUP;
            size_t vpidx = ((size_t)kv_h * LQ + s) * HD_GROUPS + vg;
            float vd = kivi4_dequant(packed_v, d, __half2float(v_scale[vpidx]), __half2float(v_zero[vpidx]));
            acc += prob * vd;
        }}
        for (int s = 0; s < RESIDUAL; ++s) {{
            float partial = 0.0f;
            const __half* ks = k_residual + ((size_t)kv_h * RESIDUAL + s) * HD;
            for (int dd = threadIdx.x; dd < HD; dd += blockDim.x)
                partial += __half2float(qh[dd]) * __half2float(ks[dd]);
            float prob = expf(block_sum(partial) * scale_attn - row_max) / denom;
            acc += prob * __half2float(v_residual[((size_t)kv_h * RESIDUAL + s) * HD + d]);
        }}
        out[(size_t)h * HD + d] = acc;
    }}
}}

"#,
        name = name
    )
}

fn kivi_attention_flash_device_kernel(name: &str) -> String {
    let unpack_fn_def = emit_kivi4_unpack4("kivi4_unpack4");
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define KIVI_CPL 4
static_assert(KIVI_GROUP == 32, "first KIVI DSL lane targets the KIVI group32 ABI");
static_assert(HD == 128, "kivi_flash maps one warp lane to four channels; HD must be 128");
static_assert(HD_PACKED * 2 == HD, "HD_PACKED must store two int4 values per byte");
static_assert(HD_GROUPS == ((HD + KIVI_GROUP - 1) / KIVI_GROUP), "bad HD_GROUPS");
static_assert(LQ_GROUPS == ((LQ + KIVI_GROUP - 1) / KIVI_GROUP), "bad LQ_GROUPS");

{unpack_fn_def}
// Single-split flash decode for the KIVI4 ABI.
//
// One warp owns one query head. Lane L owns the four contiguous channels [4L, 4L+3],
// loads two packed int4 bytes as one 16-bit word, and keeps the online-softmax value
// accumulator for those four channels in registers. This ports the vectorized `load4` +
// online-softmax schedule into the DSL lowering without yet adding split-K or
// GQA-amortized head groups.
__global__ void {name}_kivi_flash_decode(
    const __half* q,
    const unsigned char* k_code, const __half* k_scale, const __half* k_zero,
    const unsigned char* v_code, const __half* v_scale, const __half* v_zero,
    const __half* k_residual, const __half* v_residual,
    float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int lane = threadIdx.x & 31;
    int c0 = lane * KIVI_CPL;
    const __half* qh = q + (size_t)h * HD + c0;
    float qreg[KIVI_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_kg = -1;
    float ksc[KIVI_CPL] = {{1.0f, 1.0f, 1.0f, 1.0f}};
    float kz[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    for (int s = 0; s < LQ; ++s) {{
        int kg = s / KIVI_GROUP;
        if (kg != cur_kg) {{
            cur_kg = kg;
            const __half* scb = k_scale + ((size_t)kv_h * LQ_GROUPS + kg) * HD + c0;
            const __half* zb = k_zero + ((size_t)kv_h * LQ_GROUPS + kg) * HD + c0;
            #pragma unroll
            for (int i = 0; i < KIVI_CPL; ++i) {{
                ksc[i] = __half2float(scb[i]);
                kz[i] = __half2float(zb[i]);
            }}
        }}

        const unsigned char* kp = k_code + ((size_t)kv_h * LQ + s) * HD_PACKED + (c0 >> 1);
        unsigned short ktwo = (unsigned short)kp[0] | ((unsigned short)kp[1] << 8);
        float kd[KIVI_CPL];
        kivi4_unpack4(ktwo, kd, ksc, kz);

        float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
        float score = part * scale_attn;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;

        int vg = c0 / KIVI_GROUP;
        size_t vpidx = ((size_t)kv_h * LQ + s) * HD_GROUPS + vg;
        float vsc1 = __half2float(v_scale[vpidx]);
        float vz1 = __half2float(v_zero[vpidx]);
        float vsc4[KIVI_CPL] = {{vsc1, vsc1, vsc1, vsc1}};
        float vz4[KIVI_CPL] = {{vz1, vz1, vz1, vz1}};
        const unsigned char* vp = v_code + ((size_t)kv_h * LQ + s) * HD_PACKED + (c0 >> 1);
        unsigned short vtwo = (unsigned short)vp[0] | ((unsigned short)vp[1] << 8);
        float vd[KIVI_CPL];
        kivi4_unpack4(vtwo, vd, vsc4, vz4);

        #pragma unroll
        for (int i = 0; i < KIVI_CPL; ++i) acc[i] = acc[i] * corr + p * vd[i];
        m = m_new;
    }}

    for (int s = 0; s < RESIDUAL; ++s) {{
        const __half* kp = k_residual + ((size_t)kv_h * RESIDUAL + s) * HD + c0;
        float kd[KIVI_CPL] = {{
            __half2float(kp[0]), __half2float(kp[1]), __half2float(kp[2]), __half2float(kp[3])
        }};
        float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
        float score = part * scale_attn;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;

        const __half* vp = v_residual + ((size_t)kv_h * RESIDUAL + s) * HD + c0;
        #pragma unroll
        for (int i = 0; i < KIVI_CPL; ++i) acc[i] = acc[i] * corr + p * __half2float(vp[i]);
        m = m_new;
    }}

    float inv_l = 1.0f / fmaxf(l, 1.0e-30f);
    float* oh = out + (size_t)h * HD + c0;
    #pragma unroll
    for (int i = 0; i < KIVI_CPL; ++i) oh[i] = acc[i] * inv_l;
}}

"#,
        name = name,
        unpack_fn_def = unpack_fn_def,
    )
}

/// Recognized plain (non-quantized) paged GQA decode-attention ABI: uniformly fp16 K/V stored in
/// physical pages, addressed through a block table, feeding the same GQA math as the non-paged
/// exact kernel (`match_attention`'s `kv_head = head / heads_per_group`). The second real
/// consumer of `emit_gathered_page_step` -- no int4 unpack, no scale/zero, no
/// residual/quantization-boundary split (every page is uniformly fp16), and, in this first slice,
/// no runtime `SEQ_LEN`: the full static `LQ` is attended every call, unlike the KIVI paged
/// sibling.
pub(crate) struct PlainPagedAttention {
    pub(crate) nh: i64,
    pub(crate) nkv: i64,
    pub(crate) hd: i64,
    pub(crate) lq: i64,
    pub(crate) block_size: i64,
    pub(crate) num_blocks: i64,
    pub(crate) max_blocks: i64,
    /// Warps per head that each independently process an `lq / split_k` chunk of the decode
    /// loop, combined via a shared-memory flash-attention merge. `1` -- the default
    /// when a spec has no `SPLIT_K` param, so such a spec's codegen is untouched --
    /// means "no split": the original single-warp-per-head kernel.
    pub(crate) split_k: i64,
}

/// Recognize the plain paged GQA decode ABI: `Q[NH,HD]`, `BLOCK_TABLE[MAX_BLOCKS]`,
/// `K_PAGES/V_PAGES[NUM_BLOCKS,NKV,BLOCK_SIZE,HD]` (both fp16), `O[NH,HD]`. Disjoint from
/// `match_kivi_paged_attention` by tensor name alone (`K_PAGES` vs `K_CODE_PAGES` etc.) -- a spec
/// can never match both, so dispatch order against the KIVI paged matchers is not load-bearing.
pub(crate) fn match_plain_paged_attention(k: &Kernel) -> Option<PlainPagedAttention> {
    let q = tensor(k, "Q")?;
    let bt = tensor(k, "BLOCK_TABLE")?;
    let kp = tensor(k, "K_PAGES")?;
    let vp = tensor(k, "V_PAGES")?;
    let o = k.tensors.iter().find(|t| t.output && t.name == "O")?;

    let (nh, nkv, hd, lq, block_size, num_blocks, max_blocks) = (
        param(k, "NH")?,
        param(k, "NKV")?,
        param(k, "HD")?,
        param(k, "LQ")?,
        param(k, "BLOCK_SIZE")?,
        param(k, "NUM_BLOCKS")?,
        param(k, "MAX_BLOCKS")?,
    );
    // Optional: absent from every pre-existing spec, defaulting to 1 ("no split")
    // preserves their exact prior codegen output byte-for-byte.
    let split_k = param(k, "SPLIT_K").unwrap_or(1);
    if nh <= 0
        || nkv <= 0
        || hd != 128
        || lq <= 0
        || block_size <= 0
        || num_blocks <= 0
        || max_blocks <= 0
        || nh % nkv != 0
        || lq % block_size != 0
        || lq > max_blocks * block_size
        || split_k <= 0
        || lq % split_k != 0
    {
        return None;
    }
    let nh_hd = ["NH".to_string(), "HD".to_string()];
    let page_dims = [
        "NUM_BLOCKS".to_string(),
        "NKV".to_string(),
        "BLOCK_SIZE".to_string(),
        "HD".to_string(),
    ];
    if q.dims != nh_hd
        || q.dtype != "fp16"
        || o.dims != nh_hd
        || o.dtype != "fp32"
        || bt.dims != ["MAX_BLOCKS".to_string()]
        || bt.dtype != "int32"
        || kp.dims != page_dims
        || kp.dtype != "fp16"
        || vp.dims != page_dims
        || vp.dtype != "fp16"
    {
        return None;
    }

    Some(PlainPagedAttention {
        nh,
        nkv,
        hd,
        lq,
        block_size,
        num_blocks,
        max_blocks,
        split_k,
    })
}

/// The KIVI4 int4-unpack device function, shared byte-for-byte by all three KIVI decode kernels
/// below (non-paged `kivi_flash_decode`, paged single-token `kivi_flash_paged_decode`, paged
/// batched `kivi_flash_paged_decode_batched`): unpacks one 16-bit word (two int4 codes packed per
/// byte, `KIVI_CPL`=4 channels) into fp32 via each channel's own scale/zero. Parameterized by
/// name because the batched kernel is emitted into the same compiled runtime extension as the
/// paged single-token kernel and needs its own distinctly-named copy -- CUDA rejects a duplicate
/// `__device__` symbol in one translation unit.
fn emit_kivi4_unpack4(name: &str) -> String {
    format!(
        r#"__device__ __forceinline__ void {name}(unsigned short two, float out[KIVI_CPL], const float scale[KIVI_CPL], const float zero[KIVI_CPL]) {{
    out[0] = (float)(two & 0x000f) * scale[0] + zero[0];
    out[1] = (float)((two >> 4) & 0x000f) * scale[1] + zero[1];
    out[2] = (float)((two >> 8) & 0x000f) * scale[2] + zero[2];
    out[3] = (float)((two >> 12) & 0x000f) * scale[3] + zero[3];
}}
"#,
        name = name,
    )
}

/// Generic gather-indexed page-table lookup with reload-on-page-change caching: translates a
/// per-token logical index through a page table to a physical page, running `reload_body` only
/// when the physical page differs from the previous iteration's. Defines/uses the fixed local
/// names `logical_page`, `page_off`, `phys_page`, `cur_phys_page` -- callers declare
/// `cur_phys_page` before the loop (initialized to a sentinel no real page index can equal, e.g.
/// `-1`, so the first iteration always reloads) and build `reload_body` (and anything emitted
/// after this snippet) around those names. Parameterized purely by strings, no KIVI-specific
/// names: this is the reusable primitive for any future paged/gathered kernel with the same
/// reload-on-change shape (see `kivi_paged_decode_loop_body` below for the KIVI-specific caller).
fn emit_gathered_page_step(index_var: &str, table: &str, reload_body: &str) -> String {
    format!(
        r#"        int logical_page = {index_var} / BLOCK_SIZE;
        int page_off = {index_var} - logical_page * BLOCK_SIZE;
        int phys_page = {table}[logical_page];
        if (phys_page != cur_phys_page) {{
            cur_phys_page = phys_page;
{reload_body}        }}
"#,
        index_var = index_var,
        table = table,
        reload_body = reload_body,
    )
}

/// KIVI-specific paged-decode loop body shared byte-for-byte by the single-token and batched
/// paged decode kernels below: the old-quantized-pages loop (built from
/// [`emit_gathered_page_step`]) followed by the fp16 residual-window loop. The only difference
/// between the two call sites is which int4-unpack helper to call (`unpack_fn`) -- each device
/// kernel needs its own distinctly-named copy of that helper to coexist in the same compiled
/// runtime extension (see the batched kernel's module doc below).
fn kivi_paged_decode_loop_body_for(unpack_fn: &str, table_name: &str, old_limit: &str) -> String {
    // Plain text, not a format! template -- single braces here are literal CUDA, not escapes.
    let k_reload = r#"            const __half* scb = k_scale_pages + ((size_t)phys_page * NKV + kv_h) * HD + c0;
            const __half* zb = k_zero_pages + ((size_t)phys_page * NKV + kv_h) * HD + c0;
            #pragma unroll
            for (int i = 0; i < KIVI_CPL; ++i) {
                ksc[i] = __half2float(scb[i]);
                kz[i] = __half2float(zb[i]);
            }
"#;
    let gather_step = emit_gathered_page_step("s", table_name, k_reload);
    format!(
        r#"    for (int s = 0; s < {old_limit}; ++s) {{
{gather_step}
        const unsigned char* kp = k_code_pages
            + (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD_PACKED + (c0 >> 1);
        unsigned short ktwo = (unsigned short)kp[0] | ((unsigned short)kp[1] << 8);
        float kd[KIVI_CPL];
        {unpack_fn}(ktwo, kd, ksc, kz);

        float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
        float score = part * scale_attn;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;

        int vg = c0 / KIVI_GROUP;
        size_t vpidx = (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD_GROUPS + vg;
        float vsc1 = __half2float(v_scale_pages[vpidx]);
        float vz1 = __half2float(v_zero_pages[vpidx]);
        float vsc4[KIVI_CPL] = {{vsc1, vsc1, vsc1, vsc1}};
        float vz4[KIVI_CPL] = {{vz1, vz1, vz1, vz1}};
        const unsigned char* vp = v_code_pages
            + (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD_PACKED + (c0 >> 1);
        unsigned short vtwo = (unsigned short)vp[0] | ((unsigned short)vp[1] << 8);
        float vd[KIVI_CPL];
        {unpack_fn}(vtwo, vd, vsc4, vz4);

        #pragma unroll
        for (int i = 0; i < KIVI_CPL; ++i) acc[i] = acc[i] * corr + p * vd[i];
        m = m_new;
    }}

    for (int s = 0; s < residual_valid; ++s) {{
        const __half* kp = k_residual_scratch + ((size_t)kv_h * (RESIDUAL + BLOCK_SIZE) + s) * HD + c0;
        float kd[KIVI_CPL] = {{
            __half2float(kp[0]), __half2float(kp[1]), __half2float(kp[2]), __half2float(kp[3])
        }};
        float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
        float score = part * scale_attn;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;

        const __half* vp = v_residual_scratch + ((size_t)kv_h * (RESIDUAL + BLOCK_SIZE) + s) * HD + c0;
        #pragma unroll
        for (int i = 0; i < KIVI_CPL; ++i) acc[i] = acc[i] * corr + p * __half2float(vp[i]);
        m = m_new;
    }}
"#,
        old_limit = old_limit,
        gather_step = gather_step,
        unpack_fn = unpack_fn,
    )
}

fn kivi_paged_decode_loop_body(unpack_fn: &str) -> String {
    kivi_paged_decode_loop_body_for(unpack_fn, "block_table", "old_valid")
}

fn kivi_paged_sparse_retained_decode_loop_body(unpack_fn: &str) -> String {
    kivi_paged_decode_loop_body_for(unpack_fn, "retained_block_table", "retained_old_valid")
}

fn kivi_paged_attention_flash_device_kernel(name: &str) -> String {
    let unpack_fn_def = emit_kivi4_unpack4("kivi4_unpack4");
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define KIVI_CPL 4
static_assert(KIVI_GROUP == 32, "vLLM KIVI lane requires group32");
static_assert(BLOCK_SIZE == KIVI_GROUP, "vLLM KIVI page size must align with key groups");
static_assert(HD == 128, "kivi_flash maps one warp lane to four channels; HD must be 128");
static_assert(HD_PACKED * 2 == HD, "HD_PACKED must store two int4 values per byte");
static_assert(HD_GROUPS == ((HD + KIVI_GROUP - 1) / KIVI_GROUP), "bad HD_GROUPS");
static_assert(MAX_BLOCKS * BLOCK_SIZE >= LQ, "block table does not cover quantized old tokens");

{unpack_fn_def}
// Source-vLLM paged KIVI4 flash decode.
//
// One warp owns one query head. Old quantized tokens are addressed via
// BLOCK_TABLE[logical_page] -> physical page; the recent residual window stays fp16. This is the
// first block-table-callable DSL artifact and intentionally does not claim split-K/GQA parity.
//
// This paged kernel reads the residual window from K/V_RESIDUAL_SCRATCH only.
// The non-paged KIVI variants read K_RESIDUAL/V_RESIDUAL directly; those
// parameters are intentionally absent from this kernel and jac_launch ABI.
__global__ void {name}_kivi_flash_paged_decode(
    const __half* q, const int* block_table,
    const unsigned char* k_code_pages, const __half* k_scale_pages, const __half* k_zero_pages,
    const unsigned char* v_code_pages, const __half* v_scale_pages, const __half* v_zero_pages,
    const __half* k_residual_scratch, const __half* v_residual_scratch,
    int seq_len, float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int lane = threadIdx.x & 31;
    int c0 = lane * KIVI_CPL;
    const __half* qh = q + (size_t)h * HD + c0;
    float qreg[KIVI_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_phys_page = -1;
    float ksc[KIVI_CPL] = {{1.0f, 1.0f, 1.0f, 1.0f}};
    float kz[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    int bounded_seq_len = seq_len;
    if (bounded_seq_len < 0) bounded_seq_len = 0;
    if (bounded_seq_len > LQ + RESIDUAL) bounded_seq_len = LQ + RESIDUAL;
    int quantizable = bounded_seq_len - RESIDUAL;
    if (quantizable < 0) quantizable = 0;
    int old_valid = (quantizable / BLOCK_SIZE) * BLOCK_SIZE;
    if (old_valid > LQ) old_valid = LQ;
    int residual_valid = bounded_seq_len - old_valid;
    if (residual_valid < 0) residual_valid = 0;
    if (residual_valid > RESIDUAL + BLOCK_SIZE) residual_valid = RESIDUAL + BLOCK_SIZE;

{loop_body}
    float inv_l = 1.0f / fmaxf(l, 1.0e-30f);
    float* oh = out + (size_t)h * HD + c0;
    #pragma unroll
    for (int i = 0; i < KIVI_CPL; ++i) oh[i] = acc[i] * inv_l;
}}

"#,
        name = name,
        loop_body = kivi_paged_decode_loop_body("kivi4_unpack4"),
        unpack_fn_def = unpack_fn_def,
    )
}

pub(crate) fn emit_kivi_attention_body(
    k: &Kernel,
    _a: &KiviAttention,
    variant: KernelVariant,
) -> String {
    let name = &k.name;
    let kernel = match variant {
        KernelVariant::Naive => kivi_attention_device_kernel(name),
        KernelVariant::KiviFlash => kivi_attention_flash_device_kernel(name),
        KernelVariant::Cublas | KernelVariant::Wmma | KernelVariant::PagedFlash => {
            unreachable!("rejected by dispatcher")
        }
    };
    let launch = match variant {
        KernelVariant::Naive => format!(
            "{name}_kivi_decode<<<NH, 128, 0, stream>>>(Q, K_CODE, K_SCALE, K_ZERO, V_CODE, V_SCALE, V_ZERO, K_RESIDUAL, V_RESIDUAL, O);"
        ),
        KernelVariant::KiviFlash => format!(
            "{name}_kivi_flash_decode<<<NH, 32, 0, stream>>>(Q, K_CODE, K_SCALE, K_ZERO, V_CODE, V_SCALE, V_ZERO, K_RESIDUAL, V_RESIDUAL, O);"
        ),
        KernelVariant::Cublas | KernelVariant::Wmma | KernelVariant::PagedFlash => {
            unreachable!("rejected by dispatcher")
        }
    };
    format!(
        r#"{kernel}// jac_launch: KIVI ABI inputs in spec order, then O[NH,HD], then stream.
extern "C" void jac_launch(
    const __half* Q,
    const unsigned char* K_CODE, const __half* K_SCALE, const __half* K_ZERO,
    const unsigned char* V_CODE, const __half* V_SCALE, const __half* V_ZERO,
    const __half* K_RESIDUAL, const __half* V_RESIDUAL,
    float* O, cudaStream_t stream) {{
    {launch}
    CUDA_CHECK(cudaGetLastError());
}}
"#
    )
}

pub(crate) fn emit_kivi_paged_attention_library(
    k: &Kernel,
    a: &KiviPagedAttention,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::KiviFlash {
        return Err("paged KIVI vLLM decode currently supports kivi_flash only".to_string());
    }
    let name = &k.name;
    let kernel = kivi_paged_attention_flash_device_kernel(name);
    let launch = format!(
        "{name}_kivi_flash_paged_decode<<<NH, 32, 0, stream>>>(Q, BLOCK_TABLE, K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, V_CODE_PAGES, V_SCALE_PAGES, V_ZERO_PAGES, K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, SEQ_LEN, O);"
    );
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (source-vLLM paged KIVI4 flash decode).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};
static constexpr int HD = {hd};
static constexpr int NKV = {nkv};
static constexpr int LQ = {lq};
static constexpr int RESIDUAL = {residual};
static constexpr int KIVI_GROUP = {group};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int HD_GROUPS = {hd_groups};

{kernel}// jac_launch: source-vLLM paged KIVI ABI, then O[NH,HD], then stream.
// No K_RESIDUAL/V_RESIDUAL in this ABI (unused by the paged decode kernel --
// see kivi_flash_paged_decode's comment). Callers passing the older,
// 14-argument ABI (with K_RESIDUAL/V_RESIDUAL) need to update to this one.
extern "C" void jac_launch(
    const __half* Q, const int* BLOCK_TABLE,
    const unsigned char* K_CODE_PAGES, const __half* K_SCALE_PAGES, const __half* K_ZERO_PAGES,
    const unsigned char* V_CODE_PAGES, const __half* V_SCALE_PAGES, const __half* V_ZERO_PAGES,
    const __half* K_RESIDUAL_SCRATCH, const __half* V_RESIDUAL_SCRATCH,
    int SEQ_LEN, float* O, cudaStream_t stream) {{
    {launch}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        lq = a.lq,
        residual = a.residual,
        group = a.group,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        hd_packed = a.hd_packed,
        hd_groups = a.hd_groups,
        kernel = kernel,
        launch = launch,
    ))
}

fn kivi_paged_sparse_retained_attention_flash_device_kernel(name: &str) -> String {
    let unpack_fn_def = emit_kivi4_unpack4("kivi4_unpack4_sparse_retained");
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define KIVI_CPL 4
static_assert(KIVI_GROUP == 32, "vLLM KIVI lane requires group32");
static_assert(BLOCK_SIZE == KIVI_GROUP, "vLLM KIVI page size must align with key groups");
static_assert(HD == 128, "kivi_flash maps one warp lane to four channels; HD must be 128");
static_assert(HD_PACKED * 2 == HD, "HD_PACKED must store two int4 values per byte");
static_assert(HD_GROUPS == ((HD + KIVI_GROUP - 1) / KIVI_GROUP), "bad HD_GROUPS");
static_assert(MAX_BLOCKS * BLOCK_SIZE >= LQ, "retained block table does not cover quantized old tokens");

{unpack_fn_def}
// Sparse-retained source-vLLM paged KIVI4 flash decode.
//
// RETAINED_BLOCK_TABLE is compact: entry i is the physical page id for the i-th retained old page
// after page-aware eviction. The loop below scans only NUM_RETAINED_PAGES * BLOCK_SIZE old tokens.
// Residual length is still derived from the dense SEQ_LEN boundary, not from the retained-token
// count, so evicted old pages are skipped rather than accidentally treated as residual tokens.
__global__ void {name}_kivi_flash_sparse_retained_decode(
    const __half* q, const int* retained_block_table,
    const unsigned char* k_code_pages, const __half* k_scale_pages, const __half* k_zero_pages,
    const unsigned char* v_code_pages, const __half* v_scale_pages, const __half* v_zero_pages,
    const __half* k_residual_scratch, const __half* v_residual_scratch,
    int seq_len, int num_retained_pages, float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int lane = threadIdx.x & 31;
    int c0 = lane * KIVI_CPL;
    const __half* qh = q + (size_t)h * HD + c0;
    float qreg[KIVI_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_phys_page = -1;
    float ksc[KIVI_CPL] = {{1.0f, 1.0f, 1.0f, 1.0f}};
    float kz[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    int bounded_seq_len = seq_len;
    if (bounded_seq_len < 0) bounded_seq_len = 0;
    if (bounded_seq_len > LQ + RESIDUAL) bounded_seq_len = LQ + RESIDUAL;
    int quantizable = bounded_seq_len - RESIDUAL;
    if (quantizable < 0) quantizable = 0;
    int dense_old_valid = (quantizable / BLOCK_SIZE) * BLOCK_SIZE;
    if (dense_old_valid > LQ) dense_old_valid = LQ;
    int dense_old_pages = dense_old_valid / BLOCK_SIZE;
    int retained_pages = num_retained_pages;
    if (retained_pages < 0) retained_pages = 0;
    if (retained_pages > dense_old_pages) retained_pages = dense_old_pages;
    if (retained_pages > MAX_BLOCKS) retained_pages = MAX_BLOCKS;
    int retained_old_valid = retained_pages * BLOCK_SIZE;
    int residual_valid = bounded_seq_len - dense_old_valid;
    if (residual_valid < 0) residual_valid = 0;
    if (residual_valid > RESIDUAL + BLOCK_SIZE) residual_valid = RESIDUAL + BLOCK_SIZE;

{loop_body}
    float inv_l = 1.0f / fmaxf(l, 1.0e-30f);
    float* oh = out + (size_t)h * HD + c0;
    #pragma unroll
    for (int i = 0; i < KIVI_CPL; ++i) oh[i] = acc[i] * inv_l;
}}

"#,
        name = name,
        loop_body = kivi_paged_sparse_retained_decode_loop_body("kivi4_unpack4_sparse_retained"),
        unpack_fn_def = unpack_fn_def,
    )
}

pub(crate) fn emit_kivi_paged_sparse_retained_attention_library(
    k: &Kernel,
    a: &KiviPagedSparseRetainedAttention,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::KiviFlash {
        return Err(
            "sparse-retained paged KIVI vLLM decode currently supports kivi_flash only".to_string(),
        );
    }
    let name = &k.name;
    let kernel = kivi_paged_sparse_retained_attention_flash_device_kernel(name);
    let launch = format!(
        "{name}_kivi_flash_sparse_retained_decode<<<NH, 32, 0, stream>>>(Q, RETAINED_BLOCK_TABLE, K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, V_CODE_PAGES, V_SCALE_PAGES, V_ZERO_PAGES, K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, SEQ_LEN, NUM_RETAINED_PAGES, O);"
    );
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (source-vLLM sparse-retained paged KIVI4 flash decode).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};
static constexpr int HD = {hd};
static constexpr int NKV = {nkv};
static constexpr int LQ = {lq};
static constexpr int RESIDUAL = {residual};
static constexpr int KIVI_GROUP = {group};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int HD_GROUPS = {hd_groups};

{kernel}// jac_launch: sparse-retained source-vLLM paged KIVI ABI. RETAINED_BLOCK_TABLE is a compact
// physical-page table; NUM_RETAINED_PAGES bounds the old-page loop. SEQ_LEN still describes the
// original dense request length so residual_valid remains dense-boundary-correct.
extern "C" void jac_launch_sparse_retained(
    const __half* Q, const int* RETAINED_BLOCK_TABLE,
    const unsigned char* K_CODE_PAGES, const __half* K_SCALE_PAGES, const __half* K_ZERO_PAGES,
    const unsigned char* V_CODE_PAGES, const __half* V_SCALE_PAGES, const __half* V_ZERO_PAGES,
    const __half* K_RESIDUAL_SCRATCH, const __half* V_RESIDUAL_SCRATCH,
    int SEQ_LEN, int NUM_RETAINED_PAGES, float* O, cudaStream_t stream) {{
    {launch}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        lq = a.lq,
        residual = a.residual,
        group = a.group,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        hd_packed = a.hd_packed,
        hd_groups = a.hd_groups,
        kernel = kernel,
        launch = launch,
    ))
}

/// Batched sibling of [`kivi_paged_attention_flash_device_kernel`] (batched prefill-loop fix).
///
/// Every line of the per-query online-softmax math is IDENTICAL to the single-token kernel --
/// the only changes are (1) a `q_idx = blockIdx.y` batch index, (2) `seq_len` now read from
/// `SEQ_LENS[q_idx]` instead of a scalar parameter, and (3) `q`/`out` indexed by `q_idx` in
/// addition to `h`/`c0`. This is a launch-count reduction (one 2D-grid launch covering
/// MAX_QUERIES queries instead of MAX_QUERIES sequential launches), not new attention math --
/// each query's own output is expected to be bit-identical to what the single-token kernel
/// already produces for that (head, seq_len) pair. That parity is an expectation of this
/// repository, not a completed gate: neither a GPU bitcheck nor a live serving causality/order
/// check has been run in this CPU-only repository; both are required before this kernel backs
/// a served path (see docs/GPU_VALIDATION.md).
fn kivi_paged_batched_attention_flash_device_kernel(name: &str) -> String {
    let unpack_fn_def = emit_kivi4_unpack4("kivi4_unpack4_batched");
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define KIVI_CPL 4
static_assert(KIVI_GROUP == 32, "vLLM KIVI lane requires group32");
static_assert(BLOCK_SIZE == KIVI_GROUP, "vLLM KIVI page size must align with key groups");
static_assert(HD == 128, "kivi_flash maps one warp lane to four channels; HD must be 128");
static_assert(HD_PACKED * 2 == HD, "HD_PACKED must store two int4 values per byte");
static_assert(HD_GROUPS == ((HD + KIVI_GROUP - 1) / KIVI_GROUP), "bad HD_GROUPS");
static_assert(MAX_BLOCKS * BLOCK_SIZE >= LQ, "block table does not cover quantized old tokens");

{unpack_fn_def}
// One warp owns one (query, head) pair -- grid is (NH, num_queries), num_queries <= MAX_QUERIES.
// See the module doc comment above: per-query math is an exact copy of the single-token kernel's,
// only the query-batch indexing (q_idx, SEQ_LENS lookup, Q/O leading dim) is new.
__global__ void {name}_kivi_flash_paged_decode_batched(
    const __half* q, const int* seq_lens, const int* block_table,
    const unsigned char* k_code_pages, const __half* k_scale_pages, const __half* k_zero_pages,
    const unsigned char* v_code_pages, const __half* v_scale_pages, const __half* v_zero_pages,
    const __half* k_residual_scratch, const __half* v_residual_scratch,
    float* out) {{
    int h = blockIdx.x;
    int q_idx = blockIdx.y;
    int kv_h = h / HEADS_PER_GROUP;
    int lane = threadIdx.x & 31;
    int c0 = lane * KIVI_CPL;
    const __half* qh = q + ((size_t)q_idx * NH + h) * HD + c0;
    float qreg[KIVI_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_phys_page = -1;
    float ksc[KIVI_CPL] = {{1.0f, 1.0f, 1.0f, 1.0f}};
    float kz[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[KIVI_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    int seq_len = seq_lens[q_idx];
    int bounded_seq_len = seq_len;
    if (bounded_seq_len < 0) bounded_seq_len = 0;
    if (bounded_seq_len > LQ + RESIDUAL) bounded_seq_len = LQ + RESIDUAL;
    int quantizable = bounded_seq_len - RESIDUAL;
    if (quantizable < 0) quantizable = 0;
    int old_valid = (quantizable / BLOCK_SIZE) * BLOCK_SIZE;
    if (old_valid > LQ) old_valid = LQ;
    int residual_valid = bounded_seq_len - old_valid;
    if (residual_valid < 0) residual_valid = 0;
    if (residual_valid > RESIDUAL + BLOCK_SIZE) residual_valid = RESIDUAL + BLOCK_SIZE;

{loop_body}
    float inv_l = 1.0f / fmaxf(l, 1.0e-30f);
    float* oh = out + ((size_t)q_idx * NH + h) * HD + c0;
    #pragma unroll
    for (int i = 0; i < KIVI_CPL; ++i) oh[i] = acc[i] * inv_l;
}}

"#,
        name = name,
        loop_body = kivi_paged_decode_loop_body("kivi4_unpack4_batched"),
        unpack_fn_def = unpack_fn_def,
    )
}

pub(crate) fn emit_kivi_paged_batched_attention_library(
    k: &Kernel,
    a: &KiviPagedBatchedAttention,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::KiviFlash {
        return Err(
            "batched paged KIVI vLLM decode currently supports kivi_flash only".to_string(),
        );
    }
    let name = &k.name;
    let kernel = kivi_paged_batched_attention_flash_device_kernel(name);
    let launch = format!(
        "{name}_kivi_flash_paged_decode_batched<<<dim3(NH, NUM_QUERIES), 32, 0, stream>>>(Q, SEQ_LENS, BLOCK_TABLE, K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, V_CODE_PAGES, V_SCALE_PAGES, V_ZERO_PAGES, K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, O);"
    );
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (source-vLLM BATCHED paged KIVI4 flash decode).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};
static constexpr int HD = {hd};
static constexpr int NKV = {nkv};
static constexpr int LQ = {lq};
static constexpr int RESIDUAL = {residual};
static constexpr int KIVI_GROUP = {group};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int HD_GROUPS = {hd_groups};
static constexpr int MAX_QUERIES = {max_queries};

{kernel}// jac_launch_batched: source-vLLM BATCHED paged KIVI ABI, then O[MAX_QUERIES,NH,HD], then stream.
// Named distinctly from the single-token spec's `jac_launch` (not just `jac_launch`) because both
// generated translation units are compiled into the SAME runtime extension (the batched decode is
// an opt-in alternative to the single-token path, not a replacement) -- two extern "C" symbols
// both named jac_launch would collide at link time.
// NUM_QUERIES is a runtime arg (like the single-token spec's SEQ_LEN), bounding the launch grid's
// y-dimension to <= MAX_QUERIES; callers must not exceed MAX_QUERIES queries in one call.
extern "C" void jac_launch_batched(
    const __half* Q, const int* SEQ_LENS, const int* BLOCK_TABLE,
    const unsigned char* K_CODE_PAGES, const __half* K_SCALE_PAGES, const __half* K_ZERO_PAGES,
    const unsigned char* V_CODE_PAGES, const __half* V_SCALE_PAGES, const __half* V_ZERO_PAGES,
    const __half* K_RESIDUAL_SCRATCH, const __half* V_RESIDUAL_SCRATCH,
    int NUM_QUERIES, float* O, cudaStream_t stream) {{
    {launch}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        lq = a.lq,
        residual = a.residual,
        group = a.group,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        hd_packed = a.hd_packed,
        hd_groups = a.hd_groups,
        max_queries = a.max_queries,
        kernel = kernel,
        launch = launch,
    ))
}

/// DSL-authored sibling of the hand-CUDA `KIVI_CACHE_WRITE_SOURCE` kernels
/// (`integrations/vllm_kivi/source_patch.py`, functions `pack_key_age_out_page_kernel` /
/// `pack_value_age_out_page_kernel` / `append_scratch_token_kernel` /
/// `shift_scratch_after_age_out_kernel` + host orchestrator `kivi_cache_write_launch`).
///
/// Every line of math here is a direct, deliberate transcription of that hand-CUDA source, not a
/// redesign -- the only differences are `{name}_`-prefixed kernel names (avoids a link-time symbol
/// collision when both translation units are compiled into the same runtime extension, same
/// reason `jac_launch_batched` was renamed from `jac_launch` for the decode side),
/// template substitution of the dimension constants, and one addition with no effect on any output
/// tensor value: the emitted `jac_launch_cache_write` ends with `CUDA_CHECK(cudaGetLastError())`,
/// which the hand-CUDA `kivi_cache_write_launch` omits -- added for consistency with this DSL's
/// other emitted `jac_launch*` entry points, all of which check for launch errors. This must be
/// GPU-bitchecked byte-identical against the hand-CUDA writer before it can become the runtime's
/// source of truth (DSL writer).
fn kivi_cache_write_device_kernels(name: &str) -> String {
    format!(
        r#"__device__ __forceinline__ int {name}_clamp_code(float x) {{
  int code = static_cast<int>(rintf(x));
  if (code < 0) return 0;
  if (code > 15) return 15;
  return code;
}}

__device__ __forceinline__ unsigned char {name}_pack_codes(int lo, int hi) {{
  return static_cast<unsigned char>((lo & 0x0F) | ((hi & 0x0F) << 4));
}}

__device__ __forceinline__ float {name}_load_half(const __half* p, int idx) {{
  return __half2float(p[idx]);
}}

__global__ void {name}_pack_key_age_out_page_kernel(
    unsigned char* k_code_pages, __half* k_scale_pages, __half* k_zero_pages,
    const __half* k_scratch, const int* block_table, int logical_page) {{
  const int age_out_page = block_table[logical_page];
  const int h = blockIdx.x;
  const int dp = blockIdx.y;
  const int d0 = dp * 2;
  const int d1 = d0 + 1;

  float min0 = {name}_load_half(k_scratch, (h * RESIDUAL_SCRATCH) * HD + d0);
  float max0 = min0;
  float min1 = {name}_load_half(k_scratch, (h * RESIDUAL_SCRATCH) * HD + d1);
  float max1 = min1;
  for (int t = 1; t < BLOCK_SIZE; ++t) {{
    const int base = (h * RESIDUAL_SCRATCH + t) * HD;
    const float x0 = {name}_load_half(k_scratch, base + d0);
    const float x1 = {name}_load_half(k_scratch, base + d1);
    min0 = fminf(min0, x0);
    max0 = fmaxf(max0, x0);
    min1 = fminf(min1, x1);
    max1 = fmaxf(max1, x1);
  }}
  float scale0 = (max0 - min0) / 15.0f;
  float scale1 = (max1 - min1) / 15.0f;
  if (!(scale0 > 0.0f)) scale0 = 1.0f;
  if (!(scale1 > 0.0f)) scale1 = 1.0f;

  const int k_param_base = (age_out_page * NKV + h) * HD;
  k_scale_pages[k_param_base + d0] = __float2half(scale0);
  k_zero_pages[k_param_base + d0] = __float2half(min0);
  k_scale_pages[k_param_base + d1] = __float2half(scale1);
  k_zero_pages[k_param_base + d1] = __float2half(min1);

  for (int t = 0; t < BLOCK_SIZE; ++t) {{
    const int scratch_base = (h * RESIDUAL_SCRATCH + t) * HD;
    const int code0 =
        {name}_clamp_code(({name}_load_half(k_scratch, scratch_base + d0) - min0) / scale0);
    const int code1 =
        {name}_clamp_code(({name}_load_half(k_scratch, scratch_base + d1) - min1) / scale1);
    const int out =
        ((age_out_page * NKV + h) * BLOCK_SIZE + t) * HD_PACKED + dp;
    k_code_pages[out] = {name}_pack_codes(code0, code1);
  }}
}}

__global__ void {name}_pack_value_age_out_page_kernel(
    unsigned char* v_code_pages, __half* v_scale_pages, __half* v_zero_pages,
    const __half* v_scratch, const int* block_table, int logical_page) {{
  const int age_out_page = block_table[logical_page];
  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const int g = blockIdx.z;
  const int d_base = g * BLOCK_SIZE;

  float mn = {name}_load_half(v_scratch,
                       (h * RESIDUAL_SCRATCH + t) * HD + d_base);
  float mx = mn;
  for (int c = 1; c < BLOCK_SIZE; ++c) {{
    const float x = {name}_load_half(
        v_scratch, (h * RESIDUAL_SCRATCH + t) * HD + d_base + c);
    mn = fminf(mn, x);
    mx = fmaxf(mx, x);
  }}
  float scale = (mx - mn) / 15.0f;
  if (!(scale > 0.0f)) scale = 1.0f;

  const int param = ((age_out_page * NKV + h) * BLOCK_SIZE + t) * HD_GROUPS + g;
  v_scale_pages[param] = __float2half(scale);
  v_zero_pages[param] = __float2half(mn);

  for (int c = 0; c < BLOCK_SIZE; c += 2) {{
    const int d0 = d_base + c;
    const int d1 = d0 + 1;
    const int scratch_base = (h * RESIDUAL_SCRATCH + t) * HD;
    const int code0 =
        {name}_clamp_code(({name}_load_half(v_scratch, scratch_base + d0) - mn) / scale);
    const int code1 =
        {name}_clamp_code(({name}_load_half(v_scratch, scratch_base + d1) - mn) / scale);
    const int out =
        ((age_out_page * NKV + h) * BLOCK_SIZE + t) * HD_PACKED + (d0 / 2);
    v_code_pages[out] = {name}_pack_codes(code0, code1);
  }}
}}

__global__ void {name}_append_scratch_token_kernel(
    __half* k_scratch, __half* v_scratch, const __half* new_k,
    const __half* new_v, int residual_index) {{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = NKV * HD;
  if (idx >= total) return;
  const int d = idx % HD;
  const int h = idx / HD;
  const int dst = (h * RESIDUAL_SCRATCH + residual_index) * HD + d;
  k_scratch[dst] = new_k[idx];
  v_scratch[dst] = new_v[idx];
}}

__global__ void {name}_shift_scratch_after_age_out_kernel(
    __half* k_scratch, __half* v_scratch, int residual_len) {{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = NKV * HD;
  if (idx >= total) return;
  const int d = idx % HD;
  const int h = idx / HD;
  for (int r = 0; r < residual_len; ++r) {{
    const int dst = (h * RESIDUAL_SCRATCH + r) * HD + d;
    const int src = (h * RESIDUAL_SCRATCH + r + BLOCK_SIZE) * HD + d;
    k_scratch[dst] = k_scratch[src];
    v_scratch[dst] = v_scratch[src];
  }}
}}

static inline int {name}_split_old_tokens(int seq_len) {{
  int quantizable = seq_len - RESIDUAL;
  if (quantizable < 0) return 0;
  return (quantizable / BLOCK_SIZE) * BLOCK_SIZE;
}}

"#,
        name = name
    )
}

fn kivi_cache_write_batched_append_kernel(name: &str) -> String {
    format!(
        r#"__global__ void {name}_append_scratch_tokens_batched_kernel(
    __half* k_scratch, __half* v_scratch, const __half* new_k,
    const __half* new_v, int scratch_start, int num_new) {{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_token = NKV * HD;
  const int total = num_new * per_token;
  if (idx >= total) return;
  const int tok = idx / per_token;
  const int within = idx % per_token;
  const int d = within % HD;
  const int h = within / HD;
  const int dst = (h * RESIDUAL_SCRATCH + scratch_start + tok) * HD + d;
  k_scratch[dst] = new_k[idx];
  v_scratch[dst] = new_v[idx];
}}

"#,
        name = name
    )
}

pub(crate) fn emit_kivi_cache_write_batched_library(
    k: &Kernel,
    a: &KiviCacheWriteBatched,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::KiviFlash {
        return Err("KIVI batched cache-writer currently supports kivi_flash only".to_string());
    }
    let name = &k.name;
    let kernels = kivi_cache_write_device_kernels(name);
    let batched_append = kivi_cache_write_batched_append_kernel(name);
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (source-vLLM BATCHED paged KIVI4 cache writer).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{     cudaError_t err__ = (call);     if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} }} while (0)

static constexpr int NKV = {nkv};
static constexpr int HD = {hd};
static constexpr int RESIDUAL = {residual};
static constexpr int RESIDUAL_SCRATCH = {residual_scratch};
static constexpr int KIVI_GROUP = {group};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int HD_GROUPS = {hd_groups};
static constexpr int MAX_NEW = {max_new};

{kernels}{batched_append}// jac_launch_cache_write_batched: source-vLLM paged KIVI batched cache-writer ABI.
// NEW_K/NEW_V are token-major [NUM_NEW, NKV, HD]. This mirrors the hand-CUDA
// kivi_cache_write_batched_launch sub-chunk contract: append NUM_NEW tokens, then age out at most
// one page if START_SEQ_LEN + NUM_NEW crosses the RESIDUAL boundary.
extern "C" void jac_launch_cache_write_batched(
    unsigned char* K_CODE_PAGES, __half* K_SCALE_PAGES, __half* K_ZERO_PAGES,
    unsigned char* V_CODE_PAGES, __half* V_SCALE_PAGES, __half* V_ZERO_PAGES,
    __half* K_RESIDUAL_SCRATCH, __half* V_RESIDUAL_SCRATCH,
    const __half* NEW_K, const __half* NEW_V,
    const int* BLOCK_TABLE, int START_SEQ_LEN, int NUM_NEW, cudaStream_t stream) {{
    if (NUM_NEW < 1 || NUM_NEW > MAX_NEW) {{
        fprintf(stderr, "NUM_NEW out of range for jac_launch_cache_write_batched: %d\n", NUM_NEW);
        exit(1);
    }}
    const int prev_old_tokens = {name}_split_old_tokens(START_SEQ_LEN);
    const int prev_residual_len = START_SEQ_LEN - prev_old_tokens;
    const int end_seq_len = START_SEQ_LEN + NUM_NEW;
    const int next_old_tokens = {name}_split_old_tokens(end_seq_len);
    const int next_residual_len = end_seq_len - next_old_tokens;
    const int per_token = NKV * HD;
    const int total = NUM_NEW * per_token;
    {name}_append_scratch_tokens_batched_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, NEW_K, NEW_V, prev_residual_len, NUM_NEW);
    if (next_old_tokens > prev_old_tokens) {{
        const int logical_page = prev_old_tokens / BLOCK_SIZE;
        {name}_pack_key_age_out_page_kernel<<<dim3(NKV, HD_PACKED), 1, 0, stream>>>(
            K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, K_RESIDUAL_SCRATCH,
            BLOCK_TABLE, logical_page);
        {name}_pack_value_age_out_page_kernel<<<dim3(NKV, BLOCK_SIZE, HD_GROUPS), 1, 0, stream>>>(
            V_CODE_PAGES, V_SCALE_PAGES, V_ZERO_PAGES, V_RESIDUAL_SCRATCH,
            BLOCK_TABLE, logical_page);
        const int total_shift = NKV * HD;
        {name}_shift_scratch_after_age_out_kernel<<<(total_shift + 255) / 256, 256, 0, stream>>>(
            K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, next_residual_len);
    }}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nkv = a.nkv,
        hd = a.hd,
        residual = a.residual,
        residual_scratch = a.residual_scratch,
        group = a.group,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        hd_packed = a.hd_packed,
        hd_groups = a.hd_groups,
        max_new = a.max_new,
        kernels = kernels,
        batched_append = batched_append,
    ))
}

pub(crate) fn emit_kivi_cache_write_library(
    k: &Kernel,
    a: &KiviCacheWrite,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::KiviFlash {
        return Err("KIVI cache-writer currently supports kivi_flash only".to_string());
    }
    let name = &k.name;
    let kernels = kivi_cache_write_device_kernels(name);
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (source-vLLM paged KIVI4 cache writer).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NKV = {nkv};
static constexpr int HD = {hd};
static constexpr int RESIDUAL = {residual};
static constexpr int RESIDUAL_SCRATCH = {residual_scratch};
static constexpr int KIVI_GROUP = {group};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int HD_GROUPS = {hd_groups};

{kernels}// jac_launch_cache_write: source-vLLM paged KIVI cache-writer ABI. Named distinctly from the
// decode side's jac_launch/jac_launch_batched (both compiled into the same runtime extension) --
// no SLOT_MAPPING parameter, since the hand-CUDA writer's own SLOT_MAPPING argument is validated
// by its C++ wrapper but never used for addressing (BLOCK_TABLE is authoritative; see
// kivi_cache_write_launch's own comment) and this spec has no SLOT_MAPPING tensor to thread through.
extern "C" void jac_launch_cache_write(
    unsigned char* K_CODE_PAGES, __half* K_SCALE_PAGES, __half* K_ZERO_PAGES,
    unsigned char* V_CODE_PAGES, __half* V_SCALE_PAGES, __half* V_ZERO_PAGES,
    __half* K_RESIDUAL_SCRATCH, __half* V_RESIDUAL_SCRATCH,
    const __half* NEW_K, const __half* NEW_V,
    const int* BLOCK_TABLE, int SEQ_LEN, cudaStream_t stream) {{
    const int prev_seq_len = SEQ_LEN - 1;
    const int prev_old_tokens = {name}_split_old_tokens(prev_seq_len);
    const int next_old_tokens = {name}_split_old_tokens(SEQ_LEN);
    const int prev_residual_len = prev_seq_len - prev_old_tokens;
    const int next_residual_len = SEQ_LEN - next_old_tokens;
    const int total_token = NKV * HD;
    {name}_append_scratch_token_kernel<<<(total_token + 255) / 256, 256, 0, stream>>>(
        K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, NEW_K, NEW_V, prev_residual_len);
    if (next_old_tokens > prev_old_tokens) {{
        const int logical_page = prev_old_tokens / BLOCK_SIZE;
        {name}_pack_key_age_out_page_kernel<<<dim3(NKV, HD_PACKED), 1, 0, stream>>>(
            K_CODE_PAGES, K_SCALE_PAGES, K_ZERO_PAGES, K_RESIDUAL_SCRATCH,
            BLOCK_TABLE, logical_page);
        {name}_pack_value_age_out_page_kernel<<<dim3(NKV, BLOCK_SIZE, HD_GROUPS), 1, 0, stream>>>(
            V_CODE_PAGES, V_SCALE_PAGES, V_ZERO_PAGES, V_RESIDUAL_SCRATCH,
            BLOCK_TABLE, logical_page);
        const int total_shift = NKV * HD;
        {name}_shift_scratch_after_age_out_kernel<<<(total_shift + 255) / 256, 256, 0, stream>>>(
            K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH, next_residual_len);
    }}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nkv = a.nkv,
        hd = a.hd,
        residual = a.residual,
        residual_scratch = a.residual_scratch,
        group = a.group,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        hd_packed = a.hd_packed,
        hd_groups = a.hd_groups,
        kernels = kernels,
    ))
}

pub(crate) fn emit_kivi_attention(k: &Kernel, a: &KiviAttention, variant: KernelVariant) -> String {
    let name = &k.name;
    let (kernels, label, launch) = match variant {
        KernelVariant::Naive => (
            kivi_attention_device_kernel(name),
            "KIVI4 decode attention: fused score/softmax/value over quantized old KV + fp16 residual",
            format!("{name}_kivi_decode<<<NH, 128>>>(dQ, dKC, dKS, dKZ, dVC, dVS, dVZ, dKR, dVR, dO);"),
        ),
        KernelVariant::KiviFlash => (
            kivi_attention_flash_device_kernel(name),
            "KIVI4 decode attention: online-softmax flash schedule + vectorized int4 load4",
            format!("{name}_kivi_flash_decode<<<NH, 32>>>(dQ, dKC, dKS, dKZ, dVC, dVS, dVZ, dKR, dVR, dO);"),
        ),
        KernelVariant::Cublas | KernelVariant::Wmma | KernelVariant::PagedFlash => {
            unreachable!("rejected by dispatcher")
        }
    };
    format!(
        r#"// Generated by jacq — kernel `{name}` ({label}).
// DSL first lane for the faithful KIVI ABI; no scores scratch. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};
static constexpr int HD = {hd};
static constexpr int NKV = {nkv};
static constexpr int LQ = {lq};
static constexpr int RESIDUAL = {residual};
static constexpr int KIVI_GROUP = {group};
static constexpr int HD_PACKED = {hd_packed};
static constexpr int LQ_GROUPS = {lq_groups};
static constexpr int HD_GROUPS = {hd_groups};

{kernels}static __half* load_half(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    __half* host = (__half*)malloc(count * sizeof(__half));
    size_t rd = fread(host, sizeof(__half), count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    __half* dev;
    CUDA_CHECK(cudaMalloc(&dev, count * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(dev, host, count * sizeof(__half), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

static unsigned char* load_u8(const char* path, size_t count) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "cannot open %s\n", path); exit(1); }}
    unsigned char* host = (unsigned char*)malloc(count);
    size_t rd = fread(host, 1, count, f);
    if (rd != count) {{ fprintf(stderr, "short read %s: %zu/%zu\n", path, rd, count); exit(1); }}
    fclose(f);
    unsigned char* dev;
    CUDA_CHECK(cudaMalloc(&dev, count));
    CUDA_CHECK(cudaMemcpy(dev, host, count, cudaMemcpyHostToDevice));
    free(host);
    return dev;
}}

int main(int argc, char** argv) {{
    if (argc < 11) {{
        fprintf(stderr, "usage: %s <Q fp16> <K_CODE u8> <K_SCALE fp16> <K_ZERO fp16> <V_CODE u8> <V_SCALE fp16> <V_ZERO fp16> <K_RESIDUAL fp16> <V_RESIDUAL fp16> <O fp32 out>\n", argv[0]);
        return 1;
    }}
    __half* dQ = load_half(argv[1], (size_t)NH * HD);
    unsigned char* dKC = load_u8(argv[2], (size_t)NKV * LQ * HD_PACKED);
    __half* dKS = load_half(argv[3], (size_t)NKV * LQ_GROUPS * HD);
    __half* dKZ = load_half(argv[4], (size_t)NKV * LQ_GROUPS * HD);
    unsigned char* dVC = load_u8(argv[5], (size_t)NKV * LQ * HD_PACKED);
    __half* dVS = load_half(argv[6], (size_t)NKV * LQ * HD_GROUPS);
    __half* dVZ = load_half(argv[7], (size_t)NKV * LQ * HD_GROUPS);
    __half* dKR = load_half(argv[8], (size_t)NKV * RESIDUAL * HD);
    __half* dVR = load_half(argv[9], (size_t)NKV * RESIDUAL * HD);
    float* dO;
    CUDA_CHECK(cudaMalloc(&dO, (size_t)NH * HD * sizeof(float)));

    {launch}
    CUDA_CHECK(cudaGetLastError());

    float* hO = (float*)malloc((size_t)NH * HD * sizeof(float));
    CUDA_CHECK(cudaMemcpy(hO, dO, (size_t)NH * HD * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* out = fopen(argv[10], "wb");
    if (!out) {{ fprintf(stderr, "cannot write %s\n", argv[10]); return 1; }}
    fwrite(hO, sizeof(float), (size_t)NH * HD, out);
    fclose(out);
    printf("O[0:8] =");
    for (int i = 0; i < 8 && i < NH * HD; ++i) printf(" %.4f", hO[i]);
    printf("\n");
    return 0;
}}
"#,
        name = name,
        label = label,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        lq = a.lq,
        residual = a.residual,
        group = a.group,
        hd_packed = a.hd_packed,
        lq_groups = a.lq_groups,
        hd_groups = a.hd_groups,
        kernels = kernels,
        launch = launch,
    )
}

// ---- Shared utilities -----------------------------------------------------------

/// Look up a `params` value by dimension name.
/// Plain (non-quantized) paged-decode loop body: the second real consumer of
/// `emit_gathered_page_step`, proving the primitive generalizes beyond KIVI. Every page is
/// uniformly fp16 -- there is no per-page scale/zero to amortize the way KIVI's `reload_body`
/// does, so `reload_body` here is empty; only the primitive's per-iteration `phys_page`/
/// `page_off` computation is actually reused, not its reload-skip caching benefit.
///
/// `lower`/`upper` bound the loop as literal CUDA expressions (e.g. `"0"`/`"LQ"`, or
/// `"s_begin"`/`"s_end"`) so the identical recurrence serves both the K=1 kernel (the full
/// `[0, LQ)` range, one warp/head) and the split-K kernel (each of `SPLIT_K` warps/head
/// covers its own `[s_begin, s_end)` chunk) without duplicating the gather/score/update logic.
/// Still attends every token in its given range every call -- no runtime `SEQ_LEN` in this first
/// slice, unlike the KIVI paged sibling.
fn plain_paged_decode_loop_body(lower: &str, upper: &str) -> String {
    let gather_step = emit_gathered_page_step("s", "block_table", "");
    format!(
        r#"    for (int s = {lower}; s < {upper}; ++s) {{
{gather_step}
        const __half* kp = k_pages
            + (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD + c0;
        float kd[PAGED_CPL] = {{
            __half2float(kp[0]), __half2float(kp[1]), __half2float(kp[2]), __half2float(kp[3])
        }};
        float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
        float score = part * scale_attn;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * corr + p;

        const __half* vp = v_pages
            + (((size_t)phys_page * NKV + kv_h) * BLOCK_SIZE + page_off) * HD + c0;
        #pragma unroll
        for (int i = 0; i < PAGED_CPL; ++i) acc[i] = acc[i] * corr + p * __half2float(vp[i]);
        m = m_new;
    }}
"#,
        gather_step = gather_step,
        lower = lower,
        upper = upper,
    )
}

/// Plain (non-quantized) paged GQA flash-decode device kernel: one warp per query head, online
/// softmax, gathered fp16 K/V pages via `BLOCK_TABLE`, arbitrary GQA ratio (any `NH % NKV == 0`).
/// The gather-indexed sibling of `kivi_paged_attention_flash_device_kernel`, with no quantization
/// at all -- see `plain_paged_decode_loop_body`'s doc for why `reload_body` is empty here.
fn plain_paged_attention_device_kernel(name: &str) -> String {
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define PAGED_CPL 4
static_assert(HD == 128, "plain_paged_flash maps one warp lane to four channels; HD must be 128");
static_assert(LQ % BLOCK_SIZE == 0, "first slice requires whole pages only, no partial last page");
static_assert(MAX_BLOCKS * BLOCK_SIZE >= LQ, "block table does not cover the full context");

// Plain paged GQA flash decode: online-softmax attention over fp16 K/V pages gathered via
// BLOCK_TABLE, arbitrary GQA ratio (any NH % NKV == 0). No quantization, no dequant.
__global__ void {name}_plain_paged_flash_decode(
    const __half* q, const int* block_table, const __half* k_pages, const __half* v_pages,
    float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int lane = threadIdx.x & 31;
    int c0 = lane * PAGED_CPL;
    const __half* qh = q + (size_t)h * HD + c0;
    float qreg[PAGED_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_phys_page = -1;
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[PAGED_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

{loop_body}
    float inv_l = 1.0f / fmaxf(l, 1.0e-30f);
    float* oh = out + (size_t)h * HD + c0;
    #pragma unroll
    for (int i = 0; i < PAGED_CPL; ++i) oh[i] = acc[i] * inv_l;
}}

"#,
        name = name,
        loop_body = plain_paged_decode_loop_body("0", "LQ"),
    )
}

/// Split-K variant of the plain paged GQA decode kernel: `split_k` warps per head
/// instead of one, each independently processing an `LQ / split_k`-sized chunk via the identical
/// gathered-page recurrence (`plain_paged_decode_loop_body`), then combined via a shared-memory
/// flash-attention log-sum-exp merge. Algorithmic parity with the K=1 kernel is expected
/// within float64 reassociation tolerance, not bit-identical (summing in a different order
/// cannot be, by construction), and requires a GPU bitcheck to establish (see
/// docs/GPU_VALIDATION.md).
///
/// Motivation: `plain_paged_attention_device_kernel` launches `<<<NH, 32>>>` -- at
/// NH=28 that is 28 warps total for the whole kernel, a tiny fraction of a modern GPU's
/// warp-level capacity, and every one of those 28 SMs runs only a single warp with nothing else
/// resident to hide a memory load or the shuffle-reduction's dependency chain behind. That
/// under-occupancy is an architectural argument, not a measured number in this repository. This
/// variant raises the
/// warp count to `NH * split_k` without changing the ABI (same tensors, same `jac_launch`
/// signature) -- only the launch's block dimension and the kernel body change.
fn plain_paged_attention_splitk_device_kernel(name: &str, split_k: i64) -> String {
    format!(
        r#"#define HEADS_PER_GROUP (NH / NKV)
#define PAGED_CPL 4
#define SPLIT_K {split_k}
static_assert(HD == 128, "plain_paged_flash maps one warp lane to four channels; HD must be 128");
static_assert(LQ % BLOCK_SIZE == 0, "first slice requires whole pages only, no partial last page");
static_assert(MAX_BLOCKS * BLOCK_SIZE >= LQ, "block table does not cover the full context");
static_assert(LQ % SPLIT_K == 0, "split-K requires LQ to divide evenly into SPLIT_K chunks");
static_assert(SPLIT_K > 1, "SPLIT_K == 1 should use plain_paged_attention_device_kernel instead");

// Plain paged GQA flash decode, split-K: SPLIT_K warps per head, each owning an
// independent LQ/SPLIT_K chunk of the online-softmax recurrence, combined via shared memory.
// See plain_paged_attention_device_kernel (SPLIT_K=1) for the single-warp-per-head sibling this
// generalizes -- the SPLIT_K=1 kernel under-occupies a modern GPU at typical NH.
__global__ void {name}_plain_paged_flash_decode(
    const __half* q, const int* block_table, const __half* k_pages, const __half* v_pages,
    float* out) {{
    int h = blockIdx.x;
    int kv_h = h / HEADS_PER_GROUP;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int c0 = lane * PAGED_CPL;
    const __half* qh = q + (size_t)h * HD + c0;
    float qreg[PAGED_CPL] = {{
        __half2float(qh[0]), __half2float(qh[1]), __half2float(qh[2]), __half2float(qh[3])
    }};
    float scale_attn = rsqrtf((float)HD);

    int cur_phys_page = -1;
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[PAGED_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    constexpr int CHUNK = LQ / SPLIT_K;
    int s_begin = warp * CHUNK;
    int s_end = s_begin + CHUNK;

{loop_body}
    // Stage this warp's partial (m, l, acc) into shared memory, then combine on warp 0 -- a
    // small (SPLIT_K-sized) reduction, negligible next to the main loop's cost.
    __shared__ float sh_m[SPLIT_K];
    __shared__ float sh_l[SPLIT_K];
    __shared__ float sh_acc[SPLIT_K][HD];
    if (lane == 0) {{
        sh_m[warp] = m;
        sh_l[warp] = l;
    }}
    #pragma unroll
    for (int i = 0; i < PAGED_CPL; ++i) sh_acc[warp][c0 + i] = acc[i];
    __syncthreads();

    if (warp == 0) {{
        float global_m = -1.0e30f;
        #pragma unroll
        for (int w = 0; w < SPLIT_K; ++w) global_m = fmaxf(global_m, sh_m[w]);
        float global_l = 0.0f;
        float global_acc[PAGED_CPL] = {{0.0f, 0.0f, 0.0f, 0.0f}};
        #pragma unroll
        for (int w = 0; w < SPLIT_K; ++w) {{
            float corr = __expf(sh_m[w] - global_m);
            global_l += sh_l[w] * corr;
            #pragma unroll
            for (int i = 0; i < PAGED_CPL; ++i) global_acc[i] += sh_acc[w][c0 + i] * corr;
        }}
        float inv_l = 1.0f / fmaxf(global_l, 1.0e-30f);
        float* oh = out + (size_t)h * HD + c0;
        #pragma unroll
        for (int i = 0; i < PAGED_CPL; ++i) oh[i] = global_acc[i] * inv_l;
    }}
}}

"#,
        name = name,
        split_k = split_k,
        loop_body = plain_paged_decode_loop_body("s_begin", "s_end"),
    )
}

/// Library artifact for the plain paged GQA decode kernel (runtime extension embedding, no
/// standalone `main()`) -- mirrors `emit_kivi_paged_attention_library`'s assembly pattern exactly,
/// minus the quantization-specific tensors/params.
pub(crate) fn emit_plain_paged_attention_library(
    k: &Kernel,
    a: &PlainPagedAttention,
    variant: KernelVariant,
) -> Result<String, String> {
    if variant != KernelVariant::PagedFlash {
        return Err("plain paged GQA decode currently supports paged_flash only".to_string());
    }
    let name = &k.name;
    let (kernel, block_threads) = if a.split_k > 1 {
        (
            plain_paged_attention_splitk_device_kernel(name, a.split_k),
            a.split_k * 32,
        )
    } else {
        (plain_paged_attention_device_kernel(name), 32)
    };
    let launch = format!(
        "{name}_plain_paged_flash_decode<<<NH, {block_threads}, 0, stream>>>(Q, BLOCK_TABLE, K_PAGES, V_PAGES, O);"
    );
    Ok(format!(
        r#"// Generated by jacq — kernel `{name}` (plain paged GQA flash decode).
// Library artifact for embedding in a runtime extension. Regenerate; do not hand-edit.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {{ \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) {{ fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); exit(1); }} \
}} while (0)

static constexpr int NH = {nh};
static constexpr int HD = {hd};
static constexpr int NKV = {nkv};
static constexpr int LQ = {lq};
static constexpr int BLOCK_SIZE = {block_size};
static constexpr int NUM_BLOCKS = {num_blocks};
static constexpr int MAX_BLOCKS = {max_blocks};

{kernel}// jac_launch: plain paged GQA ABI in spec order, then O[NH,HD], then stream.
extern "C" void jac_launch(
    const __half* Q, const int* BLOCK_TABLE, const __half* K_PAGES, const __half* V_PAGES,
    float* O, cudaStream_t stream) {{
    {launch}
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        name = name,
        nh = a.nh,
        hd = a.hd,
        nkv = a.nkv,
        lq = a.lq,
        block_size = a.block_size,
        num_blocks = a.num_blocks,
        max_blocks = a.max_blocks,
        kernel = kernel,
        launch = launch,
    ))
}

/// Harness-compatible body emitter for the plain paged GQA decode kernel -- used by
/// `emit_kernel_body` (the `jacq --experiment`/`jac(paged_flash)` subject path), as opposed to
/// `emit_plain_paged_attention_library`'s standalone-preamble "library artifact" ABI. Unlike the
/// library path, tensor-dim params (NH/HD/NKV/BLOCK_SIZE/NUM_BLOCKS/MAX_BLOCKS) are NOT
/// redeclared here -- the harness's own `-D`-overridable dim-default macros already supply them
/// (redeclaring would collide with those macros, mirroring `emit_attention_body`'s exact
/// contract for NH/HD/NKV/L). `LQ` is the one exception: it is a pure param, never a tensor
/// dimension, so no harness macro covers it -- declared directly here instead.
pub(crate) fn emit_plain_paged_attention_body(k: &Kernel, a: &PlainPagedAttention) -> String {
    let name = &k.name;
    let (kernel, block_threads) = if a.split_k > 1 {
        (
            plain_paged_attention_splitk_device_kernel(name, a.split_k),
            a.split_k * 32,
        )
    } else {
        (plain_paged_attention_device_kernel(name), 32)
    };
    format!(
        r#"static constexpr int LQ = {lq};
{kernel}
// jac_launch: plain paged GQA ABI in spec order, then O[NH,HD], then stream.
extern "C" void jac_launch(
    const __half* Q, const int* BLOCK_TABLE, const __half* K_PAGES, const __half* V_PAGES,
    float* O, cudaStream_t stream) {{
    {name}_plain_paged_flash_decode<<<NH, {block_threads}, 0, stream>>>(Q, BLOCK_TABLE, K_PAGES, V_PAGES, O);
    CUDA_CHECK(cudaGetLastError());
}}
"#,
        lq = a.lq,
        kernel = kernel,
        name = name,
        block_threads = block_threads,
    )
}

pub(crate) fn param(k: &Kernel, name: &str) -> Option<i64> {
    k.params.iter().find(|(p, _)| p == name).map(|(_, v)| *v)
}

/// The first two-operand einsum's subscript string in a kernel's phases.
pub(crate) fn einsum_subscripts(k: &Kernel) -> Option<&str> {
    k.phases.iter().flat_map(|p| &p.body).find_map(|s| match s {
        KStmt::Einsum {
            subscripts,
            operands,
            ..
        } if operands.len() == 2 => Some(subscripts.as_str()),
        _ => None,
    })
}

/// True for a two-operand matrix-vector einsum: one index summed away, e.g. `n, tn -> t`.
/// (A vector `a`, a matrix `ba`, output `b`; `a` is contracted.)
pub(crate) fn is_matvec_subscripts(s: &str) -> bool {
    let Some((lhs, rhs)) = s.split_once("->") else {
        return false;
    };
    let ops: Vec<&str> = lhs.split(',').map(str::trim).collect();
    let out = rhs.trim();
    if ops.len() != 2 || out.len() != 1 {
        return false;
    }
    let (a, b) = (ops[0].trim(), ops[1].trim()); // vector, matrix subscripts
    a.len() == 1 && b.len() == 2 && b.contains(out) && b.contains(a) && a != out
}

/// True for a two-matrix, single-contraction einsum, e.g. `ab, bc -> ac`.
/// (`A[a,b] · B[b,c] -> C[a,c]`; `b` is contracted.)
pub(crate) fn is_matmul_subscripts(s: &str) -> bool {
    let Some((lhs, rhs)) = s.split_once("->") else {
        return false;
    };
    let ops: Vec<&str> = lhs.split(',').map(str::trim).collect();
    let out = rhs.trim();
    if ops.len() != 2 || ops[0].len() != 2 || ops[1].len() != 2 || out.len() != 2 {
        return false;
    }
    let (a, b) = (ops[0], ops[1]);
    let shared: Vec<char> = a.chars().filter(|c| b.contains(*c)).collect();
    if shared.len() != 1 {
        return false;
    }
    let n = shared[0];
    let (Some(a_out), Some(b_out)) = (a.chars().find(|&c| c != n), b.chars().find(|&c| c != n))
    else {
        return false;
    };
    !out.contains(n) && out.chars().eq([a_out, b_out])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proves `emit_gathered_page_step` is a genuinely generic primitive, not a disguised
    /// KIVI-specific helper: non-KIVI index/table names and a non-KIVI reload body round-trip
    /// through it correctly, and no KIVI identifier leaks in from the primitive's own template.
    #[test]
    fn emit_gathered_page_step_is_parameterized_not_kivi_specific() {
        let step = emit_gathered_page_step(
            "tok",
            "page_lut",
            "            widget = reload_widget(phys_page);\n",
        );
        assert!(step.contains("int logical_page = tok / BLOCK_SIZE;"));
        assert!(step.contains("int page_off = tok - logical_page * BLOCK_SIZE;"));
        assert!(step.contains("int phys_page = page_lut[logical_page];"));
        assert!(step.contains("if (phys_page != cur_phys_page) {"));
        assert!(step.contains("cur_phys_page = phys_page;"));
        assert!(step.contains("widget = reload_widget(phys_page);"));
        assert!(!step.to_lowercase().contains("kivi"));
        assert!(!step.contains("k_scale_pages"));
    }

    /// The step's reload gate is a single well-formed `if` block: opens once, closes once, and
    /// the reload body sits strictly inside it (catches an unbalanced-brace regression that
    /// string-level `contains` checks alone would miss).
    #[test]
    fn emit_gathered_page_step_reload_body_is_balanced_and_scoped() {
        let step = emit_gathered_page_step("i", "lut", "            x = 1;\n");
        assert_eq!(step.matches('{').count(), 1);
        assert_eq!(step.matches('}').count(), 1);
        let open = step.find('{').unwrap();
        let close = step.find('}').unwrap();
        let body_pos = step.find("x = 1;").unwrap();
        assert!(open < body_pos && body_pos < close);
    }

    /// `emit_gathered_page_step` must remain correct with an EMPTY reload body -- the case a
    /// caller with nothing per-page to cache (uniformly fp16 pages, no scale/zero) actually
    /// needs. Every prior test/caller always passed a non-empty reload body; this is a genuinely
    /// new edge case for the second (plain paged GQA) consumer.
    #[test]
    fn emit_gathered_page_step_works_with_empty_reload_body() {
        let step = emit_gathered_page_step("s", "block_table", "");
        assert!(step.contains("int logical_page = s / BLOCK_SIZE;"));
        assert!(step.contains("int page_off = s - logical_page * BLOCK_SIZE;"));
        assert!(step.contains("int phys_page = block_table[logical_page];"));
        assert!(step.contains("if (phys_page != cur_phys_page) {"));
        assert!(step.contains("cur_phys_page = phys_page;"));
        // Balanced even with nothing inside the reload body.
        assert_eq!(step.matches('{').count(), 1);
        assert_eq!(step.matches('}').count(), 1);
    }

    /// `emit_kivi4_unpack4` must reproduce, byte-for-byte, the int4-unpack device function
    /// previously hand-duplicated three times across the non-paged, paged single-token, and
    /// paged batched KIVI decode kernels -- this is CUDA source golden-snapshotted elsewhere, so
    /// exact text (not just "contains the right math") is the bar.
    #[test]
    fn emit_kivi4_unpack4_produces_expected_unpack_body() {
        let unpack = emit_kivi4_unpack4("kivi4_unpack4");
        // Raw string, not `\`-continued lines: a trailing `\` line-continuation in a normal Rust
        // string literal strips leading whitespace on the next line, which would silently eat
        // the 4-space indentation this test exists to lock down.
        assert_eq!(
            unpack,
            r#"__device__ __forceinline__ void kivi4_unpack4(unsigned short two, float out[KIVI_CPL], const float scale[KIVI_CPL], const float zero[KIVI_CPL]) {
    out[0] = (float)(two & 0x000f) * scale[0] + zero[0];
    out[1] = (float)((two >> 4) & 0x000f) * scale[1] + zero[1];
    out[2] = (float)((two >> 8) & 0x000f) * scale[2] + zero[2];
    out[3] = (float)((two >> 12) & 0x000f) * scale[3] + zero[3];
}
"#
        );
    }

    /// Proves the function name is a real parameter, not hardcoded -- the batched paged-decode
    /// kernel needs its own distinctly-named copy (`kivi4_unpack4_batched`) to coexist with the
    /// single-token paged kernel's `kivi4_unpack4` in one compiled runtime extension (CUDA
    /// rejects a duplicate `__device__` symbol in one translation unit).
    #[test]
    fn emit_kivi4_unpack4_is_parameterized_by_name() {
        let batched = emit_kivi4_unpack4("kivi4_unpack4_batched");
        assert!(batched.starts_with(
            "__device__ __forceinline__ void kivi4_unpack4_batched(unsigned short two, "
        ));
        // No bare-name leak alongside the batched one (the substring check requires the `(`
        // immediately after the name, so "kivi4_unpack4_batched(" does not falsely match).
        assert!(!batched.contains("void kivi4_unpack4("));
        // Body math is identical regardless of name.
        assert!(batched.contains("out[0] = (float)(two & 0x000f) * scale[0] + zero[0];"));
        assert!(batched.contains("out[3] = (float)((two >> 12) & 0x000f) * scale[3] + zero[3];"));
    }

    /// `plain_paged_decode_loop_body` is the second real, non-KIVI consumer of
    /// `emit_gathered_page_step`: proves the primitive generalizes without forcing KIVI
    /// concepts (scale/zero reload, int4 unpack, residual split) into a caller that has none of
    /// them.
    #[test]
    fn plain_paged_decode_loop_body_reuses_gather_primitive_without_kivi_leakage() {
        let body = plain_paged_decode_loop_body("0", "LQ");
        // Reuses the shared primitive's exact gather/indexing lines.
        assert!(body.contains("int logical_page = s / BLOCK_SIZE;"));
        assert!(body.contains("int phys_page = block_table[logical_page];"));
        assert!(body.contains("if (phys_page != cur_phys_page) {"));
        // Reads K/V directly from the gathered page -- nothing per-page is cached (there is
        // nothing to cache for uniformly-fp16 pages), unlike KIVI's scale/zero reload.
        assert!(body.contains("k_pages"));
        assert!(body.contains("v_pages"));
        // Attends the full static LQ every call (no runtime SEQ_LEN in this first slice).
        assert!(body.contains("for (int s = 0; s < LQ; ++s)"));
        // Zero KIVI-specific leakage: no quantization, no unpack, no scale/zero pages, no
        // residual split. (Not checking bare "scale"/"zero" -- scale_attn, the generic 1/sqrt(HD)
        // attention scale, legitimately contains "scale" and is not KIVI-specific.)
        assert!(!body.to_lowercase().contains("kivi"));
        assert!(!body.contains("unpack"));
        assert!(!body.contains("scale_pages"));
        assert!(!body.contains("zero_pages"));
        assert!(!body.contains("residual"));
        assert!(!body.contains("old_valid"));
    }

    /// The full plain-paged device kernel: GQA head-mapping and warp-reduction structure present,
    /// kernel name correctly parameterized, zero KIVI-specific leakage.
    #[test]
    fn plain_paged_attention_device_kernel_is_structurally_sound() {
        let cu = plain_paged_attention_device_kernel("test_kernel");
        assert!(cu.contains("#define HEADS_PER_GROUP (NH / NKV)"));
        assert!(cu.contains("int kv_h = h / HEADS_PER_GROUP;"));
        assert!(cu.contains("__global__ void test_kernel_plain_paged_flash_decode("));
        assert!(cu.contains("__shfl_xor_sync"));
        assert!(cu.contains("rsqrtf((float)HD)"));
        assert!(!cu.to_lowercase().contains("kivi"));
        assert!(!cu.contains("unpack"));
    }

    /// `plain_paged_decode_loop_body` must actually use its `lower`/`upper` arguments to bound
    /// the loop -- not silently ignore them and always emit `0`/`LQ` (which would make the
    /// split-K kernel's per-warp chunking a no-op, every warp redundantly scanning the same
    /// full range instead of its own slice).
    #[test]
    fn plain_paged_decode_loop_body_uses_its_given_bounds() {
        let body = plain_paged_decode_loop_body("s_begin", "s_end");
        assert!(body.contains("for (int s = s_begin; s < s_end; ++s)"));
        assert!(!body.contains("for (int s = 0; s < LQ; ++s)"));
    }

    /// The split-K device kernel: SPLIT_K warps/head, shared-memory combine present,
    /// reuses the same gather primitive as the SPLIT_K=1 kernel (no duplicated recurrence), zero
    /// KIVI leakage.
    #[test]
    fn plain_paged_attention_splitk_device_kernel_is_structurally_sound() {
        let cu = plain_paged_attention_splitk_device_kernel("test_kernel", 4);
        assert!(cu.contains("#define SPLIT_K 4"));
        assert!(cu.contains("#define HEADS_PER_GROUP (NH / NKV)"));
        assert!(cu.contains("__global__ void test_kernel_plain_paged_flash_decode("));
        assert!(cu.contains("int warp = threadIdx.x >> 5;"));
        assert!(cu.contains("constexpr int CHUNK = LQ / SPLIT_K;"));
        // Reuses the shared gather/online-softmax recurrence -- same lines the SPLIT_K=1 kernel
        // uses, bounded to this warp's own chunk instead of the full [0, LQ) range.
        assert!(cu.contains("for (int s = s_begin; s < s_end; ++s)"));
        assert!(cu.contains("int logical_page = s / BLOCK_SIZE;"));
        assert!(cu.contains("__shfl_xor_sync"));
        // Combine step: shared memory staging + a warp-0-only merge.
        assert!(cu.contains("__shared__ float sh_acc[SPLIT_K][HD];"));
        assert!(cu.contains("__syncthreads();"));
        assert!(cu.contains("if (warp == 0) {"));
        assert!(cu.contains("global_l += sh_l[w] * corr;"));
        assert!(!cu.to_lowercase().contains("kivi"));
        assert!(!cu.contains("unpack"));
    }
    #[test]
    fn sparse_retained_kivi_loop_uses_compact_table_and_sparse_bound() {
        let body = kivi_paged_sparse_retained_decode_loop_body("kivi4_unpack4_sparse_retained");
        assert!(body.contains("for (int s = 0; s < retained_old_valid; ++s)"));
        assert!(body.contains("int phys_page = retained_block_table[logical_page];"));
        assert!(!body.contains("int phys_page = block_table[logical_page];"));
    }

    #[test]
    fn sparse_retained_kivi_kernel_keeps_residual_dense_boundary() {
        let cu = kivi_paged_sparse_retained_attention_flash_device_kernel("test_sparse");
        assert!(cu.contains("__global__ void test_sparse_kivi_flash_sparse_retained_decode("));
        assert!(cu.contains("int dense_old_valid = (quantizable / BLOCK_SIZE) * BLOCK_SIZE;"));
        assert!(cu.contains("int retained_old_valid = retained_pages * BLOCK_SIZE;"));
        assert!(cu.contains("int residual_valid = bounded_seq_len - dense_old_valid;"));
        assert!(!cu.contains("int residual_valid = bounded_seq_len - retained_old_valid;"));
        assert!(cu.contains("int phys_page = retained_block_table[logical_page];"));
    }
}
