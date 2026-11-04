//! CUDA code generation for the Jacquard **kernel** DSL.
//!
//! Step 3 emits a correct, launchable CUDA kernel + a host harness for the matrix-vector (GEMV)
//! pattern — the canonical minimal kernel. A GEMV kernel computes `y[t] = Σ_n x[n] · W[t,n]` from
//! a phase whose body loads a tile of the matrix and contracts it with the vector via
//! `einsum("n, tn -> t", x, W_tile)`. This is a naive (one-thread-per-output-row, fp32-accumulate)
//! kernel; tiling / `cp.async` / WMMA / cuBLAS are optimization *variants* for a later step.

use jac_syntax::ast::Kernel;

pub use crate::variants::KernelVariant;
use crate::variants::{
    check_wmma_dims, emit_attention, emit_attention_body, emit_gemv, emit_gemv_body,
    emit_kivi_attention, emit_kivi_attention_body, emit_kivi_cache_write_batched_library,
    emit_kivi_cache_write_library, emit_kivi_paged_attention_library,
    emit_kivi_paged_batched_attention_library, emit_kivi_paged_sparse_retained_attention_library,
    emit_matmul, emit_matmul_body, emit_plain_paged_attention_body,
    emit_plain_paged_attention_library, emit_softmax, emit_softmax_body, match_attention,
    match_gemv, match_kivi_attention, match_kivi_cache_write, match_kivi_cache_write_batched,
    match_kivi_paged_attention, match_kivi_paged_batched_attention,
    match_kivi_paged_sparse_retained_attention, match_matmul, match_plain_paged_attention,
    match_softmax,
};

/// Generate a `.cu` (kernel + host harness) for a kernel spec, or describe why it is unsupported.
/// Dispatches over the recognized patterns: GEMV, matmul, and row-wise softmax.
///
/// For `KernelVariant::Wmma`, only the matmul pattern is accepted; GEMV and softmax are rejected
/// with a clear error (WMMA requires 16×16 tiles that don't fit a vector or softmax shape).
/// Additionally, M/N/K must all be multiples of 16 — enforced via [`check_wmma_dims`].
pub fn emit_kernel(k: &Kernel, variant: KernelVariant) -> Result<String, String> {
    if let Some(g) = match_gemv(k) {
        if variant == KernelVariant::Wmma {
            return Err(
                "wmma variant supports matmul only; use --naive or --cublas for GEMV".to_string(),
            );
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_gemv(k, &g, variant));
    }
    if let Some(m) = match_matmul(k) {
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        if variant == KernelVariant::Wmma {
            check_wmma_dims(&m)?;
        }
        return Ok(emit_matmul(k, &m, variant));
    }
    if let Some(a) = match_attention(k) {
        if variant == KernelVariant::Wmma {
            return Err("wmma variant supports matmul only; use --naive for attention".to_string());
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_attention(k, &a));
    }
    if let Some(a) = match_kivi_attention(k) {
        if variant == KernelVariant::Wmma || variant == KernelVariant::Cublas {
            return Err(
                "KIVI int4 decode attention supports jac naive and kivi_flash only".to_string(),
            );
        }
        return Ok(emit_kivi_attention(k, &a, variant));
    }
    if let Some(s) = match_softmax(k) {
        if variant == KernelVariant::Wmma {
            return Err("wmma variant supports matmul only; use --naive for softmax".to_string());
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_softmax(k, &s));
    }
    Err(
        "kernel codegen supports the GEMV, matmul, and row-wise softmax patterns; this kernel \
         matched none (check tensor ranks, the einsum/softmax statement, and params)"
            .to_string(),
    )
}

/// Generate a complete CUDA translation unit for embedding in a runtime extension.
///
/// Unlike [`emit_kernel`], this omits the standalone file-IO `main()` harness.  Unlike
/// [`emit_kernel_body`], it includes headers, the `CUDA_CHECK` macro, and concrete dimension
/// constants from the kernel spec, so a source-built runtime such as the KIVI vLLM fork can add the
/// generated `.cu` file directly to its CMake extension target.
///
/// The first supported library artifact is the DSL-lowered KIVI decode kernel.  Other kernel
/// patterns should be added deliberately once their runtime ABI is clear rather than by stripping a
/// benchmark harness and pretending it is production-callable.
pub fn emit_kernel_library(k: &Kernel, variant: KernelVariant) -> Result<String, String> {
    // Checked before match_kivi_paged_attention: the batched spec's Q/O tensors carry an extra
    // leading MAX_QUERIES dimension and a SEQ_LENS tensor the single-token matcher doesn't expect,
    // so the two specs can never both match the same kernel -- order between them is not
    // load-bearing for correctness, only for which error message a malformed spec sees first.
    if let Some(a) = match_kivi_paged_batched_attention(k) {
        return emit_kivi_paged_batched_attention_library(k, &a, variant);
    }
    if let Some(a) = match_kivi_paged_sparse_retained_attention(k) {
        return emit_kivi_paged_sparse_retained_attention_library(k, &a, variant);
    }
    if let Some(a) = match_kivi_paged_attention(k) {
        return emit_kivi_paged_attention_library(k, &a, variant);
    }
    // Disjoint from both KIVI paged matchers by tensor name alone (K_PAGES vs K_CODE_PAGES) --
    // order relative to them is not load-bearing, only grouped with its paged-decode siblings.
    if let Some(a) = match_plain_paged_attention(k) {
        return emit_plain_paged_attention_library(k, &a, variant);
    }
    // Checked before the single-token cache writer: the batched writer's NEW_K/NEW_V tensors carry
    // a leading MAX_NEW dimension, so the two writer specs are disjoint by shape.
    if let Some(a) = match_kivi_cache_write_batched(k) {
        return emit_kivi_cache_write_batched_library(k, &a, variant);
    }
    // Checked before match_kivi_attention: the cache-writer spec has no Q/O tensors at all (it is
    // not an attention kernel), so match_kivi_attention could never match it -- order here is not
    // load-bearing, only grouped with its paged decode siblings for readability.
    if let Some(a) = match_kivi_cache_write(k) {
        return emit_kivi_cache_write_library(k, &a, variant);
    }
    if let Some(a) = match_kivi_attention(k) {
        if variant == KernelVariant::Wmma || variant == KernelVariant::Cublas {
            return Err(
                "KIVI int4 decode attention supports jac naive and kivi_flash only".to_string(),
            );
        }
        let label = match variant {
            KernelVariant::Naive => {
                "KIVI4 decode attention: fused score/softmax/value over quantized old KV + fp16 residual"
            }
            KernelVariant::KiviFlash => {
                "KIVI4 decode attention: online-softmax flash schedule + vectorized int4 load4"
            }
            KernelVariant::Cublas | KernelVariant::Wmma | KernelVariant::PagedFlash => {
                unreachable!("rejected above")
            }
        };
        let body = emit_kivi_attention_body(k, &a, variant);
        return Ok(format!(
            r#"// Generated by jacq — kernel `{name}` ({label}).
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
static constexpr int HD_PACKED = {hd_packed};
static constexpr int LQ_GROUPS = {lq_groups};
static constexpr int HD_GROUPS = {hd_groups};

{body}"#,
            name = k.name,
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
            body = body
        ));
    }
    Err(
        "kernel library emission currently supports the KIVI int4 decode-attention ABI and the \
         plain paged GQA decode-attention ABI only"
            .to_string(),
    )
}

/// Generate the device kernel(s) + a stream-aware `extern "C" void jac_launch(...)` for a
/// kernel spec. The standalone `main()` harness is NOT included — use `emit_kernel` for that.
///
/// Emits ONLY the `__global__` device kernel(s) and the `jac_launch` wrapper — no preamble.
/// The including translation unit (the experiment harness `harness.cu`) must provide:
///   - `#include <cuda_fp16.h>` and `#include <cuda_runtime.h>`
///   - a `CUDA_CHECK(call)` macro
///   - dimension names (`M`, `N`, `K`, `R`, `C`, …) as preprocessor macros (via `#define`),
///     so that `jacq bench -D M=16384` can override them without rewriting this file.
///
/// `jac_launch` parameter order: non-output tensors in their declaration order, then the output
/// tensor, then `cudaStream_t stream`. This matches the order `emit_subject_harness` assumes.
pub(crate) fn emit_kernel_body(k: &Kernel, variant: KernelVariant) -> Result<String, String> {
    if let Some(g) = match_gemv(k) {
        if variant == KernelVariant::Wmma {
            return Err(
                "wmma variant supports matmul only; use naive or cublas for GEMV".to_string(),
            );
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_gemv_body(k, &g, variant));
    }
    if let Some(mm) = match_matmul(k) {
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        if variant == KernelVariant::Wmma {
            check_wmma_dims(&mm)?;
        }
        return Ok(emit_matmul_body(k, &mm, variant));
    }
    if let Some(a) = match_attention(k) {
        if variant == KernelVariant::Wmma {
            return Err("wmma variant supports matmul only; use naive for attention".to_string());
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_attention_body(k, &a));
    }
    if let Some(a) = match_kivi_attention(k) {
        if variant == KernelVariant::Wmma || variant == KernelVariant::Cublas {
            return Err(
                "KIVI int4 decode attention supports jac naive and kivi_flash only".to_string(),
            );
        }
        return Ok(emit_kivi_attention_body(k, &a, variant));
    }
    // Disjoint from match_kivi_attention by tensor name (K_PAGES vs K) -- order not load-bearing,
    // only grouped with its paged-decode sibling for readability (mirrors emit_kernel_library).
    if let Some(a) = match_plain_paged_attention(k) {
        if variant != KernelVariant::PagedFlash {
            return Err("plain paged GQA decode currently supports paged_flash only".to_string());
        }
        return Ok(emit_plain_paged_attention_body(k, &a));
    }
    if let Some(s) = match_softmax(k) {
        if variant == KernelVariant::Wmma {
            return Err("wmma variant supports matmul only; use naive for softmax".to_string());
        }
        if variant == KernelVariant::KiviFlash {
            return Err("kivi_flash variant supports KIVI decode attention only".to_string());
        }
        return Ok(emit_softmax_body(k, &s));
    }
    Err(
        "kernel codegen supports the GEMV, matmul, and row-wise softmax patterns; this kernel \
         matched none (check tensor ranks, the einsum/softmax statement, and params)"
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jac_syntax::parse_kernel;

    const GEMV: &str = r#"kernel gemv {
      tensors { x[N] : fp16 @registers  W[M, N] : fp16 @global -> stream  y[M] : fp32 @registers -> global }
      params { M = 4096  N = 128 }
      phase compute over W.dim(M) by M { load W_tile = W[tile, *]  y_tile = einsum("n, tn -> t", x, W_tile) }
    }"#;

    #[test]
    fn emits_gemv_kernel() {
        let k = parse_kernel(GEMV).unwrap();
        let cu = emit_kernel(&k, KernelVariant::Naive).unwrap();
        assert!(cu.contains("static constexpr int M = 4096;"));
        assert!(cu.contains("static constexpr int N = 128;"));
        assert!(cu.contains("__global__ void gemv_kernel("));
        assert!(cu.contains("acc += __half2float(W[(size_t)m * N + n]) * __half2float(x[n]);"));
        assert!(cu.contains("int main("));
    }

    #[test]
    fn emits_gemv_cublas_variant() {
        let k = parse_kernel(GEMV).unwrap();
        let cu = emit_kernel(&k, KernelVariant::Cublas).unwrap();
        assert!(cu.contains("#include <cublas_v2.h>"));
        assert!(cu.contains("cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N,"));
        assert!(cu.contains("CUDA_R_32F, M, CUBLAS_COMPUTE_32F"));
        assert!(
            !cu.contains("__global__"),
            "cuBLAS variant should not define a hand-written kernel"
        );
    }

    #[test]
    fn rejects_non_gemv() {
        // No 2-D matrix tensor → not a GEMV.
        let k = parse_kernel(
            r#"kernel k { tensors { a[N] : fp16 @registers  b[N] : fp32 @registers -> global }
               phase p { b_t = einsum("n -> n", a) } }"#,
        )
        .unwrap();
        assert!(emit_kernel(&k, KernelVariant::Naive).is_err());
    }

    const MATMUL: &str = r#"kernel matmul {
      tensors { A[M, N] : fp16 @global -> stream  B[N, K] : fp16 @global -> stream  C[M, K] : fp32 @registers -> global }
      params { M = 256  N = 128  K = 64 }
      phase compute over A.dim(M) by M { load A_tile = A[tile, *]  C_tile = einsum("mn, nk -> mk", A_tile, B) }
    }"#;

    #[test]
    fn emits_matmul_kernel() {
        let k = parse_kernel(MATMUL).unwrap();
        let cu = emit_kernel(&k, KernelVariant::Naive).unwrap();
        assert!(cu.contains("static constexpr int M = 256;"));
        assert!(cu.contains("static constexpr int N = 128;"));
        assert!(cu.contains("static constexpr int K = 64;"));
        assert!(cu
            .contains("__global__ void matmul_kernel(const __half* A, const __half* B, float* C)"));
        assert!(cu.contains(
            "acc += __half2float(A[(size_t)row * N + n]) * __half2float(B[(size_t)n * K + col]);"
        ));
    }

    #[test]
    fn emits_matmul_cublas_variant() {
        let k = parse_kernel(MATMUL).unwrap();
        let cu = emit_kernel(&k, KernelVariant::Cublas).unwrap();
        assert!(cu.contains("cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,"));
        assert!(!cu.contains("__global__"));
    }

    #[test]
    fn emits_softmax_kernel() {
        let k = parse_kernel(
            r#"kernel sm {
              tensors { x[R, C] : fp16 @global -> stream  y[R, C] : fp32 @registers -> global }
              params { R = 64  C = 128 }
              phase compute over x.dim(R) by R { y = softmax(x, dim: C) }
            }"#,
        )
        .unwrap();
        let cu = emit_kernel(&k, KernelVariant::Naive).unwrap();
        assert!(cu.contains("static constexpr int R = 64;"));
        assert!(cu.contains("static constexpr int C = 128;"));
        assert!(cu.contains("__global__ void sm_kernel(const __half* x, float* y)"));
        assert!(
            cu.contains("expf(__half2float(xr[i]) - rmax)"),
            "stable softmax (subtract row max)"
        );
        assert!(cu.contains("<<<R, 256>>>"));
    }

    #[test]
    fn matmul_subscript_recognition() {
        use crate::variants::is_matmul_subscripts;
        assert!(is_matmul_subscripts("ab, bc -> ac"));
        assert!(is_matmul_subscripts("mn, nk -> mk"));
        assert!(!is_matmul_subscripts("n, tn -> t")); // matvec, not matmul
        assert!(!is_matmul_subscripts("ab, bc -> ca")); // wrong output order
    }

    #[test]
    fn matvec_subscript_recognition() {
        use crate::variants::is_matvec_subscripts;
        assert!(is_matvec_subscripts("n, tn -> t"));
        assert!(is_matvec_subscripts("d, hd -> h"));
        assert!(!is_matvec_subscripts("n, n -> n")); // two vectors, no matrix
        assert!(!is_matvec_subscripts("ab, bc -> ac")); // matmul, not matvec
    }

    // ---- WMMA tests ---------------------------------------------------------------

    /// A matmul spec with M/N/K all multiples of 16 — the valid WMMA case.
    const MATMUL_16: &str = r#"kernel matmul {
      tensors { A[M, N] : fp16 @global -> stream  B[N, K] : fp16 @global -> stream  C[M, K] : fp32 @registers -> global }
      params { M = 256  N = 256  K = 256 }
      phase compute over A.dim(M) by M { load A_tile = A[tile, *]  C_tile = einsum("mn, nk -> mk", A_tile, B) }
    }"#;

    #[test]
    fn emits_matmul_wmma_variant() {
        let k = parse_kernel(MATMUL_16).unwrap();
        let cu = emit_kernel(&k, KernelVariant::Wmma).unwrap();
        // Must contain nvcuda::wmma idioms.
        assert!(
            cu.contains("wmma::fragment"),
            "missing wmma::fragment declaration"
        );
        assert!(cu.contains("mma_sync"), "missing mma_sync call");
        assert!(
            cu.contains("store_matrix_sync"),
            "missing store_matrix_sync"
        );
        assert!(cu.contains("__half"), "missing __half type");
        // fp32 accumulator.
        assert!(
            cu.contains("wmma::accumulator"),
            "missing fp32 accumulator fragment"
        );
        assert!(
            cu.contains("fill_fragment(frag_acc, 0.0f)"),
            "missing zero-fill accumulator"
        );
        // Standalone path includes the right headers.
        assert!(cu.contains("#include <mma.h>"), "missing mma.h include");
        assert!(
            cu.contains("using namespace nvcuda"),
            "missing nvcuda namespace"
        );
        // Dim constants are present.
        assert!(cu.contains("static constexpr int M = 256;"));
        assert!(cu.contains("static constexpr int N = 256;"));
        assert!(cu.contains("static constexpr int K = 256;"));
        // Must have a main() harness (standalone path).
        assert!(cu.contains("int main("));
    }

    #[test]
    fn wmma_rejects_non_multiple_of_16_m() {
        // M = 15 → not a multiple of 16.
        let k = parse_kernel(r#"kernel matmul {
          tensors { A[M, N] : fp16 @global -> stream  B[N, K] : fp16 @global -> stream  C[M, K] : fp32 @registers -> global }
          params { M = 15  N = 16  K = 16 }
          phase compute over A.dim(M) by M { load A_tile = A[tile, *]  C_tile = einsum("mn, nk -> mk", A_tile, B) }
        }"#).unwrap();
        let err = emit_kernel(&k, KernelVariant::Wmma).unwrap_err();
        assert!(
            err.contains("wmma requires M/N/K multiples of 16"),
            "error should name the constraint; got: {err}"
        );
        assert!(
            err.contains("M=15") || err.contains("15"),
            "error should mention the bad value; got: {err}"
        );
    }

    #[test]
    fn wmma_rejects_non_multiple_of_16_k() {
        // K = 33 → not a multiple of 16.
        let k = parse_kernel(r#"kernel matmul {
          tensors { A[M, N] : fp16 @global -> stream  B[N, K] : fp16 @global -> stream  C[M, K] : fp32 @registers -> global }
          params { M = 32  N = 16  K = 33 }
          phase compute over A.dim(M) by M { load A_tile = A[tile, *]  C_tile = einsum("mn, nk -> mk", A_tile, B) }
        }"#).unwrap();
        let err = emit_kernel(&k, KernelVariant::Wmma).unwrap_err();
        assert!(
            err.contains("wmma requires M/N/K multiples of 16"),
            "error should name the constraint; got: {err}"
        );
        assert!(
            err.contains("K=33") || err.contains("33"),
            "error should mention the bad value; got: {err}"
        );
    }

    #[test]
    fn wmma_rejects_gemv() {
        // WMMA does not support GEMV (vector output) — should return a clear Err.
        let k = parse_kernel(GEMV).unwrap();
        let err = emit_kernel(&k, KernelVariant::Wmma).unwrap_err();
        assert!(
            err.contains("wmma variant supports matmul only"),
            "error should explain wmma is matmul-only; got: {err}"
        );
    }

    // ---- Attention ----------------------------------------------------------------

    const ATTENTION: &str = r#"kernel decode_attention {
      tensors { Q[NH, HD] : fp16 @global -> stream  K[NKV, L, HD] : fp16 @global -> stream  V[NKV, L, HD] : fp16 @global -> stream  O[NH, HD] : fp32 @registers -> global }
      params { NH = 32  HD = 64  NKV = 4  L = 2048 }
      phase compute over Q.dim(NH) by NH { scores = einsum("hd, hsd -> hs", Q, K)  probs = softmax(scores, dim: L)  O = einsum("hs, hsd -> hd", probs, V) }
    }"#;

    #[test]
    fn matches_decode_attention() {
        use crate::variants::match_attention;
        let k = parse_kernel(ATTENTION).unwrap();
        let a = match_attention(&k).expect("decode attention should match");
        assert_eq!(a.nh, 32, "NH");
        assert_eq!(a.nkv, 4, "NKV");
        assert_eq!(a.hd, 64, "HD");
        assert_eq!(a.l, 2048, "L");
        assert_eq!(a.nh / a.nkv, 8, "heads_per_group = NH/NKV");
        assert_eq!(a.sink, None, "exact attention has no sink/window policy");
        assert_eq!(a.window, None);
    }

    const ATTENTION_WINDOWED: &str = r#"kernel decode_attention_windowed {
      tensors { Q[NH, HD] : fp16 @global -> stream  K[NKV, L, HD] : fp16 @global -> stream  V[NKV, L, HD] : fp16 @global -> stream  O[NH, HD] : fp32 @registers -> global }
      params { NH = 32  HD = 64  NKV = 4  L = 2048  SINK = 4  WINDOW = 512 }
      phase compute over Q.dim(NH) by NH { scores = einsum("hd, hsd -> hs", Q, K)  probs = softmax(scores, dim: L)  O = einsum("hs, hsd -> hd", probs, V) }
    }"#;

    #[test]
    fn matches_windowed_attention() {
        use crate::variants::match_attention;
        let win = match_attention(&parse_kernel(ATTENTION_WINDOWED).unwrap()).unwrap();
        assert_eq!(win.sink, Some(4), "SINK param captured");
        assert_eq!(win.window, Some(512), "WINDOW param captured");
        assert_eq!(win.l, 2048, "L still resolved");
    }

    #[test]
    fn emits_attention_body() {
        use crate::variants::{emit_attention_body, match_attention};
        let k = parse_kernel(ATTENTION).unwrap();
        let a = match_attention(&k).unwrap();
        let body = emit_attention_body(&k, &a);
        // GQA mapping derived from the fixture params.
        assert!(
            body.contains("#define HEADS_PER_GROUP (NH / NKV)"),
            "GQA group define"
        );
        // Three device kernels: scores -> softmax -> weighted-V, named by the kernel.
        assert!(body.contains("__global__ void decode_attention_score(const __half* q, const __half* k, float* scores)"));
        assert!(
            body.contains("rsqrtf((float)HD)"),
            "1/sqrt(HD) score scaling"
        );
        assert!(body.contains("__global__ void decode_attention_softmax(float* scores)"));
        assert!(body.contains("__global__ void decode_attention_value(float* out, const float* scores, const __half* v)"));
        // jac_launch ABI: non-output inputs in declaration order (Q, K, V), then O, then stream.
        assert!(body.contains(r#"extern "C" void jac_launch(const __half* Q, const __half* K, const __half* V, float* O, cudaStream_t stream)"#));
        // Self-managed scores scratch (cannot define jac_workspace_bytes in the harness TU).
        assert!(body.contains("cudaMalloc"), "self-managed scores scratch");
        assert!(
            body.contains("<<<NH, 128, 0, stream>>>"),
            "score/softmax launch over heads"
        );
        assert!(
            body.contains("<<<NH, 64, 0, stream>>>"),
            "value launch over heads"
        );
        // EXACT attention: no window mask in the base variant.
        assert!(
            !body.contains("INFINITY"),
            "base attention reads all L (no window mask)"
        );
    }

    #[test]
    fn emits_windowed_attention_body() {
        use crate::variants::{emit_attention_body, match_attention};
        // Windowed: the score kernel masks dropped keys via attn_kept(s).
        let kw = parse_kernel(ATTENTION_WINDOWED).unwrap();
        let aw = match_attention(&kw).unwrap();
        let win = emit_attention_body(&kw, &aw);
        assert!(
            win.contains("attn_kept"),
            "windowed kernel guards via attn_kept"
        );
        assert!(
            win.contains("(s < SINK) || (s >= L - WINDOW)"),
            "sink+window kept(s) policy"
        );
        assert!(win.contains("-INFINITY"), "dropped keys masked to -INF");
        // BUGFIX (caught on GPU): the harness only -D-defines tensor-dim params (NH/HD/NKV/L), NOT
        // SINK/WINDOW — so the generated windowed body must SELF-define the policy (like masked.cu),
        // else nvcc errors "identifier SINK is undefined". Guarded by #ifndef so -D can still override.
        assert!(
            win.contains("#ifndef SINK"),
            "self-define SINK (harness does not)"
        );
        assert!(
            win.contains("#define SINK 4"),
            "SINK value from the spec param"
        );
        assert!(
            win.contains("#ifndef WINDOW") && win.contains("#define WINDOW 512"),
            "self-define WINDOW"
        );
        // The exact variant must remain mask-free (no regression).
        let ke = parse_kernel(ATTENTION).unwrap();
        let ae = match_attention(&ke).unwrap();
        let exact = emit_attention_body(&ke, &ae);
        assert!(!exact.contains("attn_kept"), "exact attention has no mask");
        assert!(!exact.contains("INFINITY"));
    }

    #[test]
    fn emits_attention_standalone() {
        use crate::variants::{emit_attention, match_attention};
        let k = parse_kernel(ATTENTION).unwrap();
        let a = match_attention(&k).unwrap();
        let cu = emit_attention(&k, &a);
        assert!(cu.contains("#include <cuda_fp16.h>"), "standalone includes");
        assert!(cu.contains("static constexpr int NH = 32;"));
        assert!(cu.contains("static constexpr int HD = 64;"));
        assert!(cu.contains("static constexpr int NKV = 4;"));
        assert!(cu.contains("static constexpr int L = 2048;"));
        assert!(cu.contains("__global__ void decode_attention_score("));
        assert!(cu.contains("__global__ void decode_attention_softmax("));
        assert!(cu.contains("__global__ void decode_attention_value("));
        assert!(cu.contains("int main("), "standalone main harness");
        assert!(cu.contains("load_half"), "reads Q/K/V from files");
    }

    #[test]
    fn emit_kernel_dispatches_attention() {
        let k = parse_kernel(ATTENTION).unwrap();
        // Standalone path: emit_kernel routes the attention pattern to emit_attention.
        let cu = emit_kernel(&k, KernelVariant::Naive).unwrap();
        assert!(cu.contains("int main("));
        assert!(cu.contains("__global__ void decode_attention_score("));
        // Body path: emit_kernel_body routes to emit_attention_body (harness ABI).
        let body = emit_kernel_body(&k, KernelVariant::Naive).unwrap();
        assert!(body.contains(r#"extern "C" void jac_launch(const __half* Q, const __half* K, const __half* V, float* O, cudaStream_t stream)"#));
    }

    #[test]
    fn attention_rejects_wmma() {
        let k = parse_kernel(ATTENTION).unwrap();
        let err = emit_kernel(&k, KernelVariant::Wmma).unwrap_err();
        assert!(
            err.contains("wmma variant supports matmul only"),
            "attention is not a WMMA shape; got: {err}"
        );
    }

    const KIVI_ATTENTION: &str = r#"kernel decode_attention_kivi {
      tensors {
        Q[NH, HD] : fp16 @global -> stream
        K_CODE[NKV, LQ, HD_PACKED] : int8 @global -> stream
        K_SCALE[NKV, LQ_GROUPS, HD] : fp16 @global -> stream
        K_ZERO[NKV, LQ_GROUPS, HD] : fp16 @global -> stream
        V_CODE[NKV, LQ, HD_PACKED] : int8 @global -> stream
        V_SCALE[NKV, LQ, HD_GROUPS] : fp16 @global -> stream
        V_ZERO[NKV, LQ, HD_GROUPS] : fp16 @global -> stream
        K_RESIDUAL[NKV, RESIDUAL, HD] : fp16 @global -> stream
        V_RESIDUAL[NKV, RESIDUAL, HD] : fp16 @global -> stream
        O[NH, HD] : fp32 @registers -> global
      }
      params { NH = 28 NKV = 4 HD = 128 LQ = 3968 RESIDUAL = 128 KIVI_GROUP = 32 HD_PACKED = 64 LQ_GROUPS = 124 HD_GROUPS = 4 }
      phase fused_decode over Q.dim(NH) by NH { }
    }"#;

    const KIVI_PAGED_ATTENTION: &str = r#"kernel decode_attention_kivi_paged {
      tensors {
        Q[NH, HD]                                             : fp16 @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                               : int32 @global -> stream
        K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream
        K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16 @global -> stream
        K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16 @global -> stream
        V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream
        V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16 @global -> stream
        V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16 @global -> stream
        K_RESIDUAL[NKV, RESIDUAL, HD]                         : fp16 @global -> stream
        V_RESIDUAL[NKV, RESIDUAL, HD]                         : fp16 @global -> stream
        K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream
        V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream
        O[NH, HD]                                             : fp32 @registers -> global
      }
      params { NH = 40 NKV = 8 HD = 128 LQ = 3968 RESIDUAL = 128 RESIDUAL_SCRATCH = 160 KIVI_GROUP = 32 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 HD_PACKED = 64 HD_GROUPS = 4 }
      phase fused_decode over Q.dim(NH) by NH { }
    }"#;

    const KIVI_PAGED_BATCHED_ATTENTION: &str = r#"kernel decode_attention_kivi_paged_batched {
      tensors {
        Q[MAX_QUERIES, NH, HD]                                : fp16 @global -> stream
        SEQ_LENS[MAX_QUERIES]                                 : int32 @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                               : int32 @global -> stream
        K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream
        K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16 @global -> stream
        K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16 @global -> stream
        V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream
        V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16 @global -> stream
        V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16 @global -> stream
        K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream
        V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream
        O[MAX_QUERIES, NH, HD]                                : fp32 @registers -> global
      }
      params { NH = 40 NKV = 8 HD = 128 LQ = 3968 RESIDUAL = 128 RESIDUAL_SCRATCH = 160 KIVI_GROUP = 32 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 HD_PACKED = 64 HD_GROUPS = 4 MAX_QUERIES = 160 }
      phase fused_decode over Q.dim(NH) by NH { }
    }"#;

    #[test]
    fn emits_kivi_flash_paged_batched_library_artifact_for_prefill_chunks() {
        let k = parse_kernel(KIVI_PAGED_BATCHED_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(cu.contains("source-vLLM BATCHED paged KIVI4 flash decode"));
        // Distinct entry point name from the single-token spec's `jac_launch` -- both generated
        // translation units are compiled into the same runtime extension (opt-in alternative, not
        // a replacement), so identical extern "C" names would collide at link time.
        assert!(cu.contains(r#"extern "C" void jac_launch_batched("#));
        assert!(!cu.contains(r#"extern "C" void jac_launch("#));
        assert!(cu.contains("static constexpr int NH = 40;"));
        assert!(cu.contains("static constexpr int NKV = 8;"));
        assert!(cu.contains("static constexpr int MAX_QUERIES = 160;"));
        assert!(cu.contains("const int* SEQ_LENS"));
        assert!(cu.contains("const int* BLOCK_TABLE"));
        assert!(cu.contains("int q_idx = blockIdx.y;"));
        assert!(cu.contains("int seq_len = seq_lens[q_idx];"));
        assert!(cu.contains("block_table[logical_page]"));
        assert!(cu.contains("K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH"));
        assert!(cu.contains("RESIDUAL + BLOCK_SIZE"));
        assert!(cu.contains("(quantizable / BLOCK_SIZE) * BLOCK_SIZE"));
        assert!(cu.contains("for (int s = 0; s < old_valid; ++s)"));
        assert!(cu.contains("for (int s = 0; s < residual_valid; ++s)"));
        assert!(cu.contains(
            "__global__ void decode_attention_kivi_paged_batched_kivi_flash_paged_decode_batched("
        ));
        assert!(cu.contains("<<<dim3(NH, NUM_QUERIES), 32, 0, stream>>>"));
        assert!(cu.contains("int NUM_QUERIES, float* O"));
        assert!(!cu.contains("int main("));
        assert!(!cu.contains("load_half("));
    }

    #[test]
    fn kivi_flash_paged_batched_decode_per_query_math_matches_single_token_kernel() {
        // The single-token and batched kernels must compute IDENTICAL per-query math -- this is a
        // launch-count reduction, not new attention logic (see the module doc comment on
        // kivi_paged_batched_attention_flash_device_kernel). Compare the two kernel bodies'
        // arithmetic lines directly so a future edit to one that isn't mirrored to the other is
        // caught here rather than only at GPU bitcheck time.
        let single = parse_kernel(KIVI_PAGED_ATTENTION).unwrap();
        let single_cu = emit_kernel_library(&single, KernelVariant::KiviFlash).unwrap();
        let batched = parse_kernel(KIVI_PAGED_BATCHED_ATTENTION).unwrap();
        let batched_cu = emit_kernel_library(&batched, KernelVariant::KiviFlash).unwrap();

        // Lines that must appear byte-identical in both kernel bodies (the shared online-softmax
        // math, the page/residual addressing, and the quantized-code unpacking) -- anything that
        // differs between the two here is either a real batching-only change (query/seq_len
        // indexing, tested separately above) or an unintended divergence this guard should catch.
        let shared_math_lines = [
            "float scale_attn = rsqrtf((float)HD);",
            "if (bounded_seq_len < 0) bounded_seq_len = 0;",
            "if (bounded_seq_len > LQ + RESIDUAL) bounded_seq_len = LQ + RESIDUAL;",
            "int quantizable = bounded_seq_len - RESIDUAL;",
            "int old_valid = (quantizable / BLOCK_SIZE) * BLOCK_SIZE;",
            "int residual_valid = bounded_seq_len - old_valid;",
            "int logical_page = s / BLOCK_SIZE;",
            "int phys_page = block_table[logical_page];",
            "float part = qreg[0] * kd[0] + qreg[1] * kd[1] + qreg[2] * kd[2] + qreg[3] * kd[3];",
            "for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);",
            "float score = part * scale_attn;",
            "float m_new = fmaxf(m, score);",
            "float corr = __expf(m - m_new);",
            "float p = __expf(score - m_new);",
            "l = l * corr + p;",
            "acc[i] = acc[i] * corr + p * vd[i];",
            "acc[i] = acc[i] * corr + p * __half2float(vp[i]);",
            "float inv_l = 1.0f / fmaxf(l, 1.0e-30f);",
        ];
        for line in shared_math_lines {
            assert!(
                single_cu.contains(line),
                "single-token kernel missing expected shared math line: {line}"
            );
            assert!(
                batched_cu.contains(line),
                "batched kernel missing expected shared math line (diverged from single-token): {line}"
            );
        }
    }

    const KIVI_CACHE_WRITE: &str = r#"kernel cache_write_kivi_paged {
      tensors {
        NEW_K[NKV, HD]                                        : fp16 @global -> stream
        NEW_V[NKV, HD]                                        : fp16 @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                               : int32 @global -> stream
        K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream -> global
        K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                    : fp16 @global -> stream -> global
        K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                     : fp16 @global -> stream -> global
        V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED]  : int8 @global -> stream -> global
        V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16 @global -> stream -> global
        V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS]  : fp16 @global -> stream -> global
        K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream -> global
        V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]         : fp16 @global -> stream -> global
      }
      params { NKV = 8 HD = 128 RESIDUAL = 128 RESIDUAL_SCRATCH = 160 KIVI_GROUP = 32 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 HD_PACKED = 64 HD_GROUPS = 4 }
      phase cache_write over NEW_K.dim(NKV) by NKV { }
    }"#;

    const KIVI_CACHE_WRITE_BATCHED: &str = r#"kernel cache_write_kivi_paged_batched {
      tensors {
        NEW_K[MAX_NEW, NKV, HD]                             : fp16 @global -> stream
        NEW_V[MAX_NEW, NKV, HD]                             : fp16 @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                             : int32 @global -> stream
        K_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED] : int8 @global -> stream -> global
        K_SCALE_PAGES[NUM_BLOCKS, NKV, HD]                  : fp16 @global -> stream -> global
        K_ZERO_PAGES[NUM_BLOCKS, NKV, HD]                   : fp16 @global -> stream -> global
        V_CODE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_PACKED] : int8 @global -> stream -> global
        V_SCALE_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16 @global -> stream -> global
        V_ZERO_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD_GROUPS] : fp16 @global -> stream -> global
        K_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]        : fp16 @global -> stream -> global
        V_RESIDUAL_SCRATCH[NKV, RESIDUAL_SCRATCH, HD]        : fp16 @global -> stream -> global
      }
      params { NKV = 8 HD = 128 RESIDUAL = 128 RESIDUAL_SCRATCH = 160 KIVI_GROUP = 32 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 HD_PACKED = 64 HD_GROUPS = 4 MAX_NEW = 160 }
      phase cache_write_batched over NEW_K.dim(MAX_NEW) by MAX_NEW { }
    }"#;

    #[test]
    fn emits_kivi_cache_write_batched_library_artifact_for_prefill_subchunks() {
        let k = parse_kernel(KIVI_CACHE_WRITE_BATCHED).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(cu.contains("source-vLLM BATCHED paged KIVI4 cache writer"));
        assert!(cu.contains(r#"extern "C" void jac_launch_cache_write_batched("#));
        assert!(!cu.contains(r#"extern "C" void jac_launch_cache_write("#));
        assert!(!cu.contains(r#"extern "C" void jac_launch("#));
        assert!(cu.contains("static constexpr int NKV = 8;"));
        assert!(cu.contains("static constexpr int MAX_NEW = 160;"));
        assert!(cu.contains("const int* BLOCK_TABLE"));
        assert!(cu.contains("int START_SEQ_LEN, int NUM_NEW"));
        assert!(cu.contains("cache_write_kivi_paged_batched_append_scratch_tokens_batched_kernel"));
        assert!(cu.contains("const int per_token = NKV * HD;"));
        assert!(cu.contains("const int total = NUM_NEW * per_token;"));
        assert!(cu.contains("const int tok = idx / per_token;"));
        assert!(
            cu.contains("const int dst = (h * RESIDUAL_SCRATCH + scratch_start + tok) * HD + d;")
        );
        assert!(cu.contains("const int prev_old_tokens = cache_write_kivi_paged_batched_split_old_tokens(START_SEQ_LEN);"));
        assert!(cu.contains("const int end_seq_len = START_SEQ_LEN + NUM_NEW;"));
        assert!(cu.contains("if (next_old_tokens > prev_old_tokens)"));
        assert!(cu.contains("const int logical_page = prev_old_tokens / BLOCK_SIZE;"));
        assert!(cu.contains("<<<dim3(NKV, HD_PACKED), 1, 0, stream>>>"));
        assert!(cu.contains("<<<dim3(NKV, BLOCK_SIZE, HD_GROUPS), 1, 0, stream>>>"));
        assert!(!cu.contains("int main("));
    }

    #[test]
    fn kivi_cache_write_batched_reuses_single_writer_pack_and_shift_math() {
        let single = parse_kernel(KIVI_CACHE_WRITE).unwrap();
        let single_cu = emit_kernel_library(&single, KernelVariant::KiviFlash).unwrap();
        let batched = parse_kernel(KIVI_CACHE_WRITE_BATCHED).unwrap();
        let batched_cu = emit_kernel_library(&batched, KernelVariant::KiviFlash).unwrap();
        let shared_lines = [
            "int code = static_cast<int>(rintf(x));",
            "return static_cast<unsigned char>((lo & 0x0F) | ((hi & 0x0F) << 4));",
            "const int age_out_page = block_table[logical_page];",
            "float scale0 = (max0 - min0) / 15.0f;",
            "float scale1 = (max1 - min1) / 15.0f;",
            "const int param = ((age_out_page * NKV + h) * BLOCK_SIZE + t) * HD_GROUPS + g;",
            "int quantizable = seq_len - RESIDUAL;",
            "return (quantizable / BLOCK_SIZE) * BLOCK_SIZE;",
            "const int total_shift = NKV * HD;",
        ];
        for line in shared_lines {
            assert!(
                single_cu.contains(line),
                "single writer missing shared line: {line}"
            );
            assert!(
                batched_cu.contains(line),
                "batched writer diverged from shared line: {line}"
            );
        }
    }

    #[test]
    fn emits_kivi_cache_write_library_artifact_for_paged_writer() {
        let k = parse_kernel(KIVI_CACHE_WRITE).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(cu.contains("source-vLLM paged KIVI4 cache writer"));
        // Distinct entry point name from the decode side's jac_launch/jac_launch_batched -- all
        // three generated translation units are compiled into the same runtime extension.
        assert!(cu.contains(r#"extern "C" void jac_launch_cache_write("#));
        assert!(!cu.contains(r#"extern "C" void jac_launch("#));
        assert!(!cu.contains(r#"extern "C" void jac_launch_batched("#));
        assert!(cu.contains("static constexpr int NKV = 8;"));
        assert!(cu.contains("static constexpr int RESIDUAL_SCRATCH = 160;"));
        // No Q/O/SEQ_LENS/attention math at all -- this is a cache-management kernel, not decode.
        assert!(!cu.contains("scale_attn"));
        assert!(!cu.contains("__shfl_xor_sync"));
        // The four device kernels, all `{name}_`-prefixed to avoid a link-time symbol collision
        // with the hand-CUDA writer's identically-named kernels once both are compiled together.
        assert!(cu.contains("cache_write_kivi_paged_pack_key_age_out_page_kernel"));
        assert!(cu.contains("cache_write_kivi_paged_pack_value_age_out_page_kernel"));
        assert!(cu.contains("cache_write_kivi_paged_append_scratch_token_kernel"));
        assert!(cu.contains("cache_write_kivi_paged_shift_scratch_after_age_out_kernel"));
        assert!(cu.contains("<<<dim3(NKV, HD_PACKED), 1, 0, stream>>>"));
        assert!(cu.contains("<<<dim3(NKV, BLOCK_SIZE, HD_GROUPS), 1, 0, stream>>>"));
        assert!(cu.contains("if (next_old_tokens > prev_old_tokens)"));
        assert!(!cu.contains("int main("));
    }

    #[test]
    fn kivi_cache_write_math_matches_hand_cuda_source() {
        // Transcription guard: the DSL-emitted quantize/append/shift/orchestration math must stay
        // byte-identical (module constant names aside) to the hand-CUDA `KIVI_CACHE_WRITE_SOURCE`
        // in integrations/vllm_kivi/source_patch.py -- a future edit to one that isn't mirrored to
        // the other should be caught here rather than only at GPU bitcheck time (DSL writer).
        let k = parse_kernel(KIVI_CACHE_WRITE).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        let hand_cuda_lines = [
            "int code = static_cast<int>(rintf(x));",
            "return static_cast<unsigned char>((lo & 0x0F) | ((hi & 0x0F) << 4));",
            "const int age_out_page = block_table[logical_page];",
            "const int d0 = dp * 2;",
            "const int d1 = d0 + 1;",
            "float scale0 = (max0 - min0) / 15.0f;",
            "float scale1 = (max1 - min1) / 15.0f;",
            "if (!(scale0 > 0.0f)) scale0 = 1.0f;",
            "if (!(scale1 > 0.0f)) scale1 = 1.0f;",
            "const int k_param_base = (age_out_page * NKV + h) * HD;",
            "const int d_base = g * BLOCK_SIZE;",
            "float scale = (mx - mn) / 15.0f;",
            "if (!(scale > 0.0f)) scale = 1.0f;",
            "const int param = ((age_out_page * NKV + h) * BLOCK_SIZE + t) * HD_GROUPS + g;",
            "const int idx = blockIdx.x * blockDim.x + threadIdx.x;",
            "const int total = NKV * HD;",
            "if (idx >= total) return;",
            "const int d = idx % HD;",
            "const int h = idx / HD;",
            "int quantizable = seq_len - RESIDUAL;",
            "if (quantizable < 0) return 0;",
            "return (quantizable / BLOCK_SIZE) * BLOCK_SIZE;",
            "const int prev_seq_len = SEQ_LEN - 1;",
            "const int prev_residual_len = prev_seq_len - prev_old_tokens;",
            "const int next_residual_len = SEQ_LEN - next_old_tokens;",
            "const int total_token = NKV * HD;",
            "if (next_old_tokens > prev_old_tokens) {",
            "const int logical_page = prev_old_tokens / BLOCK_SIZE;",
            "const int total_shift = NKV * HD;",
        ];
        for line in hand_cuda_lines {
            assert!(
                cu.contains(line),
                "DSL-emitted cache writer missing expected hand-CUDA-matching line: {line}"
            );
        }
    }

    #[test]
    fn emits_kivi_attention_body() {
        use crate::variants::match_kivi_attention;

        let k = parse_kernel(KIVI_ATTENTION).unwrap();
        let a = match_kivi_attention(&k).expect("KIVI ABI should match");
        assert_eq!(a.nh, 28);
        assert_eq!(a.nkv, 4);
        assert_eq!(a.hd, 128);
        assert_eq!(a.lq, 3968);
        assert_eq!(a.residual, 128);

        let body = emit_kernel_body(&k, KernelVariant::Naive).unwrap();
        assert!(body.contains("KIVI_GROUP == 32"));
        assert!(body.contains("__global__ void decode_attention_kivi_kivi_decode("));
        assert!(body.contains("K_CODE, K_SCALE, K_ZERO"));
        assert!(body.contains("K_RESIDUAL, V_RESIDUAL"));
        assert!(
            !body.contains("cudaMalloc((void**)&scores"),
            "KIVI fused-decode emitter must not materialize scores scratch"
        );
    }

    #[test]
    fn emits_kivi_flash_attention_body() {
        let k = parse_kernel(KIVI_ATTENTION).unwrap();
        let body = emit_kernel_body(&k, KernelVariant::KiviFlash).unwrap();
        assert!(body.contains("__global__ void decode_attention_kivi_kivi_flash_decode("));
        assert!(
            body.contains("lane * KIVI_CPL"),
            "flash variant should map each warp lane to four contiguous channels"
        );
        assert!(
            body.contains("unsigned short ktwo"),
            "flash variant should vectorize int4 loads as two packed bytes / four nibbles"
        );
        assert!(
            body.contains("l = l * corr + p"),
            "flash variant should use online softmax"
        );
        assert!(body.contains("<<<NH, 32, 0, stream>>>"));
    }

    #[test]
    fn emits_kivi_flash_library_artifact_without_standalone_main() {
        let k = parse_kernel(KIVI_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(cu.contains("Library artifact for embedding in a runtime extension"));
        assert!(cu.contains("static constexpr int NH = 28;"));
        assert!(cu.contains("__global__ void decode_attention_kivi_kivi_flash_decode("));
        assert!(cu.contains(r#"extern "C" void jac_launch("#));
        assert!(cu.contains("<<<NH, 32, 0, stream>>>"));
        assert!(
            !cu.contains("int main("),
            "library artifact must not ship a CLI harness"
        );
        assert!(
            !cu.contains("load_half("),
            "library artifact must not do file IO"
        );
    }

    #[test]
    fn emits_kivi_flash_paged_library_artifact_for_vllm_block_tables() {
        let k = parse_kernel(KIVI_PAGED_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(cu.contains("source-vLLM paged KIVI4 flash decode"));
        assert!(cu.contains("static constexpr int NH = 40;"));
        assert!(cu.contains("static constexpr int NKV = 8;"));
        assert!(cu.contains("const int* BLOCK_TABLE"));
        assert!(cu.contains("K_CODE_PAGES"));
        assert!(cu.contains("block_table[logical_page]"));
        assert!(cu.contains("int SEQ_LEN, float* O"));
        assert!(cu.contains("K_RESIDUAL_SCRATCH, V_RESIDUAL_SCRATCH"));
        assert!(cu.contains("RESIDUAL + BLOCK_SIZE"));
        assert!(cu.contains("(quantizable / BLOCK_SIZE) * BLOCK_SIZE"));
        assert!(cu.contains("for (int s = 0; s < old_valid; ++s)"));
        assert!(cu.contains("for (int s = 0; s < residual_valid; ++s)"));
        assert!(cu.contains("__global__ void decode_attention_kivi_paged_kivi_flash_paged_decode("));
        assert!(cu.contains("<<<NH, 32, 0, stream>>>"));
        assert!(!cu.contains("int main("));
        assert!(!cu.contains("load_half("));
    }

    #[test]
    fn kivi_flash_paged_decode_has_no_unused_residual_params() {
        // The paged decode reads the residual window from scratch tensors.
        // Keep unused K_RESIDUAL/V_RESIDUAL parameters out of its kernel and
        // jac_launch ABI. The separate non-paged variants read those tensors
        // directly and have a different signature.
        let k = parse_kernel(KIVI_PAGED_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::KiviFlash).unwrap();
        assert!(!cu.contains("(void)k_residual"));
        assert!(!cu.contains("(void)v_residual"));
        assert!(!cu.contains("const __half* k_residual,"));
        assert!(!cu.contains("const __half* K_RESIDUAL,"));
        assert!(!cu.contains("K_RESIDUAL, V_RESIDUAL,"));
        // The scratch variants must still be present and used.
        assert!(cu.contains("const __half* k_residual_scratch"));
        assert!(cu.contains("k_residual_scratch +"));
    }

    // Plain (non-KIVI) paged GQA decode: the second real consumer of
    // emit_gathered_page_step, proving the gather-indexed codegen extraction
    // generalizes beyond KIVI. NH=28/NKV=4 -> GQA group_size 7, a ratio FlashInfer's
    // paged decode cannot serve natively without q-head padding.
    const PLAIN_PAGED_ATTENTION: &str = r#"kernel decode_attention_paged {
      tensors {
        Q[NH, HD]                                : fp16  @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                  : int32 @global -> stream
        K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
        V_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
        O[NH, HD]                                : fp32  @registers -> global
      }
      params { NH = 28 NKV = 4 HD = 128 LQ = 3968 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 }
      phase fused_decode over Q.dim(NH) by NH { }
    }"#;

    #[test]
    fn emits_plain_paged_flash_library_artifact_for_arbitrary_gqa_ratio() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::PagedFlash).unwrap();
        assert!(cu.contains("plain paged GQA flash decode"));
        assert!(cu.contains("static constexpr int NH = 28;"));
        assert!(cu.contains("static constexpr int NKV = 4;")); // 28/4 = group_size 7
        assert!(cu.contains("const int* BLOCK_TABLE"));
        assert!(cu.contains("K_PAGES"));
        assert!(cu.contains("V_PAGES"));
        assert!(cu.contains("block_table[logical_page]"));
        assert!(cu.contains("__global__ void decode_attention_paged_plain_paged_flash_decode("));
        assert!(cu.contains("<<<NH, 32, 0, stream>>>"));
        assert!(!cu.contains("int main("));
        assert!(!cu.to_lowercase().contains("kivi"));
    }

    #[test]
    fn plain_paged_rejects_naive_cublas_and_wmma() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION).unwrap();
        for v in [
            KernelVariant::Naive,
            KernelVariant::Cublas,
            KernelVariant::Wmma,
        ] {
            let err = emit_kernel_library(&k, v).unwrap_err();
            assert!(
                err.contains("paged_flash"),
                "unexpected error for {v:?}: {err}"
            );
        }
    }

    #[test]
    fn emits_plain_paged_flash_body_for_experiment_harness() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION).unwrap();
        let body = emit_kernel_body(&k, KernelVariant::PagedFlash).unwrap();
        assert!(body.contains("static constexpr int LQ = 3968;"));
        assert!(body.contains("__global__ void decode_attention_paged_plain_paged_flash_decode("));
        assert!(body.contains("block_table[logical_page]"));
        assert!(body.contains(r#"extern "C" void jac_launch("#));
        assert!(body.contains("<<<NH, 32, 0, stream>>>"));
        // Unlike emit_kernel_library, the harness body must NOT redeclare tensor-dim macros --
        // those come from the harness's own `-D`-overridable dim-default mechanism instead
        // (see emit_subject_harness's doc comment / test in experiment.rs). LQ is the sole
        // exception since it is a pure param, never a tensor dimension.
        assert!(!body.contains("static constexpr int NH"));
        assert!(!body.contains("static constexpr int NKV"));
        assert!(!body.contains("static constexpr int HD"));
        assert!(!body.contains("int main("));
    }

    // Split-K: same ABI as PLAIN_PAGED_ATTENTION, one extra param. The SPLIT_K=1
    // kernel launches only NH=28 warps total, under-occupying a modern GPU; this
    // raises the warp count to NH*SPLIT_K. This crate has no CPU-only .cu execution
    // harness, so these tests are structural/emission-level only, matching this file's
    // established test bar for the SPLIT_K=1 kernel above; algorithmic parity of the
    // split-K kernel with SPLIT_K=1 remains to be established by the GPU bitcheck
    // gates (see docs/GPU_VALIDATION.md).
    const PLAIN_PAGED_ATTENTION_SPLITK: &str = r#"kernel decode_attention_paged_splitk {
      tensors {
        Q[NH, HD]                                : fp16  @global -> stream
        BLOCK_TABLE[MAX_BLOCKS]                  : int32 @global -> stream
        K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
        V_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
        O[NH, HD]                                : fp32  @registers -> global
      }
      params { NH = 28 NKV = 4 HD = 128 LQ = 3968 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 SPLIT_K = 4 }
      phase fused_decode over Q.dim(NH) by NH { }
    }"#;

    #[test]
    fn emits_plain_paged_flash_splitk_library_artifact_when_split_k_param_present() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION_SPLITK).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::PagedFlash).unwrap();
        // 4 warps/head instead of 1: block dimension is SPLIT_K*32, not the SPLIT_K=1 kernel's 32.
        assert!(cu.contains("<<<NH, 128, 0, stream>>>"));
        assert!(cu.contains("#define SPLIT_K 4"));
        assert!(cu.contains("static_assert(LQ % SPLIT_K == 0"));
        // Same gather recurrence, reused (not duplicated) from the SPLIT_K=1 kernel's primitive.
        assert!(cu.contains("block_table[logical_page]"));
        // Shared-memory combine machinery must be present.
        assert!(cu.contains("__shared__ float sh_m[SPLIT_K];"));
        assert!(cu.contains("__shared__ float sh_acc[SPLIT_K][HD];"));
        assert!(cu.contains("__syncthreads();"));
        assert!(cu.contains("if (warp == 0)"));
        assert!(!cu.contains("int main("));
        assert!(!cu.to_lowercase().contains("kivi"));
    }

    #[test]
    fn emits_plain_paged_flash_splitk_body_for_experiment_harness() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION_SPLITK).unwrap();
        let body = emit_kernel_body(&k, KernelVariant::PagedFlash).unwrap();
        assert!(body.contains("static constexpr int LQ = 3968;"));
        assert!(body.contains("<<<NH, 128, 0, stream>>>"));
        assert!(body.contains("#define SPLIT_K 4"));
        assert!(!body.contains("static constexpr int NH"));
        assert!(!body.contains("int main("));
    }

    #[test]
    fn plain_paged_flash_without_split_k_param_matches_the_split_k_1_kernel() {
        // The exact SPLIT_K=1 assertions (emits_plain_paged_flash_library_artifact_for_arbitrary_gqa_ratio
        // above) already pin this; this test additionally confirms the SPLIT_K-specific markers
        // are ABSENT when the spec has no SPLIT_K param, i.e. the two code paths are genuinely
        // disjoint, not one silently subsuming the other.
        let k = parse_kernel(PLAIN_PAGED_ATTENTION).unwrap();
        let cu = emit_kernel_library(&k, KernelVariant::PagedFlash).unwrap();
        assert!(cu.contains("<<<NH, 32, 0, stream>>>"));
        assert!(!cu.contains("SPLIT_K"));
        assert!(!cu.contains("__shared__"));
        assert!(!cu.contains("__syncthreads"));
    }

    #[test]
    fn plain_paged_flash_rejects_split_k_that_does_not_divide_lq_evenly() {
        // LQ=3968 is not divisible by 3 -- match_plain_paged_attention must return None (not
        // silently truncate or round), so this falls through to the generic "matched none" error
        // rather than emitting a kernel with an incorrect/undefined chunk boundary.
        const BAD: &str = r#"kernel decode_attention_paged_splitk_bad {
          tensors {
            Q[NH, HD]                                : fp16  @global -> stream
            BLOCK_TABLE[MAX_BLOCKS]                  : int32 @global -> stream
            K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
            V_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16  @global -> stream
            O[NH, HD]                                : fp32  @registers -> global
          }
          params { NH = 28 NKV = 4 HD = 128 LQ = 3968 BLOCK_SIZE = 32 NUM_BLOCKS = 124 MAX_BLOCKS = 124 SPLIT_K = 3 }
          phase fused_decode over Q.dim(NH) by NH { }
        }"#;
        let k = parse_kernel(BAD).unwrap();
        assert!(emit_kernel_library(&k, KernelVariant::PagedFlash).is_err());
        assert!(emit_kernel_body(&k, KernelVariant::PagedFlash).is_err());
    }

    #[test]
    fn plain_paged_body_rejects_naive_cublas_and_wmma() {
        let k = parse_kernel(PLAIN_PAGED_ATTENTION).unwrap();
        for v in [
            KernelVariant::Naive,
            KernelVariant::Cublas,
            KernelVariant::Wmma,
        ] {
            let err = emit_kernel_body(&k, v).unwrap_err();
            assert!(
                err.contains("paged_flash"),
                "unexpected error for {v:?}: {err}"
            );
        }
    }

    #[test]
    fn kivi_attention_rejects_cublas_and_wmma() {
        let k = parse_kernel(KIVI_ATTENTION).unwrap();
        let cublas = emit_kernel(&k, KernelVariant::Cublas).unwrap_err();
        assert!(cublas.contains("KIVI int4 decode attention"));
        let wmma = emit_kernel(&k, KernelVariant::Wmma).unwrap_err();
        assert!(wmma.contains("KIVI int4 decode attention"));
    }
}
