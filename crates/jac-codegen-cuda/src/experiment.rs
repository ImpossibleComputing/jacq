//! Experiment harness codegen: wrap a kernel op in the jac_ctx benchmark harness.
use crate::KernelVariant;
use jac_syntax::ast::Kernel;
use jac_syntax::SubjectSrc;

const HARNESS: &str = include_str!("templates/harness.cu");

/// Emit a benchmark harness `.cu` for one subject of `op`.
/// `op` is the parsed fixture kernel (gives the ABI: input/output tensors + swept dims).
///
/// The `jac_launch` signature in the emitted device code follows tensor-declaration order:
/// all non-output tensors first (in the order they appear in `op.tensors`), then the single
/// output tensor, then `cudaStream_t stream`. The `jac_run` wrapper calls `jac_launch` in
/// this same order using `ctx->inputs[i]` in declaration order and `ctx->outputs[0]`.
pub fn emit_subject_harness(op: &Kernel, src: &SubjectSrc) -> Result<String, String> {
    // ABI: inputs (non-output tensors in declaration order) then the single output tensor.
    let inputs: Vec<&jac_syntax::ast::TensorDecl> =
        op.tensors.iter().filter(|t| !t.output).collect();
    let n_outputs = op.tensors.iter().filter(|t| t.output).count();
    if n_outputs > 1 {
        return Err(format!(
            "emit_subject_harness supports exactly one `-> global` output tensor; op `{}` has {n_outputs}",
            op.name
        ));
    }
    let output = op
        .tensors
        .iter()
        .find(|t| t.output)
        .ok_or_else(|| "op has no `-> global` output tensor".to_string())?;

    let ctype = |dt: &str| match dt {
        "fp16" => "__half",
        "fp32" => "float",
        "bf16" => "__nv_bfloat16",
        // Kernel fixtures use `int8` as an 8-bit carrier. KIVI int4 packed codes are byte
        // payloads (0..255), so use unsigned char rather than signed char for generated ABIs.
        "int8" => "unsigned char",
        "int32" => "int",
        _ => "float",
    };

    // Dim defaults: every dim name used by any tensor becomes a -D-overridable macro from params.
    let mut dims: Vec<String> = Vec::new();
    for t in &op.tensors {
        for d in &t.dims {
            if !dims.contains(d) {
                dims.push(d.clone());
            }
        }
    }
    let dim_defaults: String = dims
        .iter()
        .map(|d| {
            let v = op
                .params
                .iter()
                .find(|(p, _)| p == d)
                .map(|(_, v)| *v)
                .unwrap_or(0);
            format!("#ifndef {d}\n#define {d} {v}\n#endif\n")
        })
        .collect();

    // __ALLOC__: declare/fill each input (seeded), malloc output, set ctx arrays + sizes.
    let elems = |t: &jac_syntax::ast::TensorDecl| t.dims.join(" * ");
    let mut alloc = String::new();
    let mut total: Vec<String> = Vec::new();
    for (i, t) in inputs.iter().enumerate() {
        let cty = ctype(&t.dtype);
        let n = elems(t);
        if t.name == "BLOCK_TABLE" {
            // A gather index table needs valid, in-range, genuinely-varied entries. The
            // generic float-in-[-1,1)-cast-to-int scheme below truncates EVERY element to
            // exactly 0 for an int tensor (never exercising the gather indirection at all --
            // weaker than even an identity table). Seed a Fisher-Yates permutation of
            // [0, NUM_BLOCKS) instead (the macro is already in scope: NUM_BLOCKS is a dim of
            // any page tensor a BLOCK_TABLE indexes into, so __DIM_DEFAULTS__ always defines
            // it), cycled to fill BLOCK_TABLE's own length.
            alloc.push_str(&format!(
                "    {{ size_t ne = (size_t)({n}); const int nb = NUM_BLOCKS; \
                      std::vector<{cty}> perm(nb); for (int p = 0; p < nb; ++p) perm[p] = p; \
                      for (int p = nb - 1; p > 0; --p) {{ \
                          seed = seed*6364136223846793005ULL+1442695040888963407ULL; \
                          double frac = (seed>>11)*(1.0/9007199254740992.0); \
                          int j = (int)(frac * (p + 1)); \
                          std::swap(perm[p], perm[j]); }} \
                      std::vector<{cty}> h(ne); for (size_t k=0;k<ne;k++) h[k]=perm[k % (size_t)nb]; \
                      void* d; CUDA_CHECK(cudaMalloc(&d, ne*sizeof({cty}))); \
                      CUDA_CHECK(cudaMemcpyAsync(d,h.data(),ne*sizeof({cty}),cudaMemcpyHostToDevice,stream)); \
                      inputs[{i}]=d; }}\n"
            ));
            total.push(format!("(size_t)({n})*sizeof({cty})"));
            continue;
        }
        alloc.push_str(&format!(
            "    {{ size_t ne = (size_t)({n}); std::vector<{cty}> h(ne); \
                  for (size_t k=0;k<ne;k++){{ seed = seed*6364136223846793005ULL+1442695040888963407ULL; \
                  double r=((seed>>11)*(1.0/9007199254740992.0))*2.0-1.0; h[k]=({cty})r; }} \
                  void* d; CUDA_CHECK(cudaMalloc(&d, ne*sizeof({cty}))); \
                  CUDA_CHECK(cudaMemcpyAsync(d,h.data(),ne*sizeof({cty}),cudaMemcpyHostToDevice,stream)); \
                  inputs[{i}]=d; }}\n"
        ));
        total.push(format!("(size_t)({n})*sizeof({cty})"));
    }
    let octy = ctype(&output.dtype);
    let on = elems(output);
    alloc.push_str(&format!(
        "    {{ size_t ne=(size_t)({on}); void* d; CUDA_CHECK(cudaMalloc(&d, ne*sizeof({octy}))); \
              outputs[0]=d; out_elems[0]=ne; out_elem_size[0]=sizeof({octy}); }}\n"
    ));
    total.push(format!("(size_t)({on})*sizeof({octy})"));
    let alloc = format!("    n_in = {}; n_out = 1;\n{}", inputs.len(), alloc);
    let alloc_bytes = format!("alloc_bytes = {};", total.join(" + "));

    // __SUBJECT_IMPL__: either #include the user .cu, or emit the jac kernel body + jac_run wrapper.
    let subject_impl = match src {
        SubjectSrc::Cuda(path) => format!("#include \"{path}\""),
        SubjectSrc::Jac(variant) => {
            let kvar = match variant.as_str() {
                "naive" => KernelVariant::Naive,
                "cublas" => KernelVariant::Cublas,
                "wmma" => KernelVariant::Wmma,
                "kivi_flash" | "kivi-flash" => KernelVariant::KiviFlash,
                "paged_flash" | "paged-flash" => KernelVariant::PagedFlash,
                other => {
                    return Err(format!(
                        "unknown jac variant `{other}` (valid: naive, cublas, wmma, kivi_flash, paged_flash)"
                    ))
                }
            };
            let body = crate::kernel::emit_kernel_body(op, kvar)?;
            // jac_launch takes inputs in declaration order, then output, then stream.
            // This matches the order `emit_kernel_body` uses for the `jac_launch` parameters.
            let calls = inputs
                .iter()
                .enumerate()
                .map(|(i, t)| format!("({}*)ctx->inputs[{i}]", ctype(&t.dtype)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{body}\nextern \"C\" void jac_run(jac_ctx* ctx) {{ \
                     jac_launch({calls}, ({}*)ctx->outputs[0], ctx->stream); }}",
                ctype(&output.dtype)
            )
        }
    };

    // Extra top-of-harness includes for variants whose body code relies on them.
    // The wmma body emits `#include <mma.h>` and `using namespace nvcuda;` inline, so no extra
    // include is needed here — but cublas_v2.h must come before the subject body is included.
    let extra_includes = if matches!(src, SubjectSrc::Jac(v) if v == "cublas") {
        "#include <cublas_v2.h>\n"
    } else {
        ""
    };

    Ok(HARNESS
        .replace("__NAME__", &op.name)
        .replace("__INCLUDES__", extra_includes)
        .replace("__DIM_DEFAULTS__", &dim_defaults)
        .replace("__ALLOC__", &alloc)
        .replace("__ALLOC_BYTES__", &alloc_bytes)
        .replace("__SUBJECT_IMPL__", &subject_impl))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jac_syntax::parse_kernel;

    const GEMV: &str = r#"kernel gemv {
      tensors { x[N] : fp16 @registers  W[M, N] : fp16 @global -> stream  y[M] : fp32 @registers -> global }
      params { M = 4096  N = 128 }
      phase compute over W.dim(M) by M { load t = W[tile, *]  y_tile = einsum("n, tn -> t", x, W_tile) }
    }"#;

    #[test]
    fn emits_jac_subject_harness() {
        let op = parse_kernel(GEMV).unwrap();
        let cu = emit_subject_harness(&op, &SubjectSrc::Jac("naive".into())).unwrap();
        assert!(cu.contains("struct jac_ctx"));
        assert!(cu.contains("extern \"C\" void jac_run(jac_ctx* ctx)"));
        assert!(cu.contains("#ifndef M")); // dim defaults overridable by -D
        assert!(cu.contains("#ifndef N"));
        assert!(cu.contains(r#"\"backend\":\"events\""#));
        assert!(!cu.contains("__ALLOC__")); // all placeholders substituted
        assert!(!cu.contains("__SUBJECT_IMPL__"));

        // the injected subject body must NOT redeclare dims as constants (would collide with the -D macros)
        assert!(
            !cu.contains("constexpr int M"),
            "subject body must use the harness M macro, not redeclare it"
        );
        assert!(
            !cu.contains("constexpr int N"),
            "subject body must use the harness N macro, not redeclare it"
        );
        // exactly one CUDA_CHECK definition and one of each CUDA include (no duplicate preamble from the body)
        assert_eq!(
            cu.matches("#define CUDA_CHECK").count(),
            1,
            "duplicate CUDA_CHECK from subject preamble"
        );
        assert_eq!(
            cu.matches("#include <cuda_runtime.h>").count(),
            1,
            "duplicate cuda_runtime.h include from subject preamble"
        );
        assert_eq!(
            cu.matches("#include <cuda_fp16.h>").count(),
            1,
            "duplicate cuda_fp16.h include from subject preamble"
        );
        // the -D-overridable dim macro mechanism must be intact
        assert!(
            cu.contains("#ifndef M"),
            "harness #ifndef M dim-default block must be present"
        );
        assert!(
            cu.contains("#ifndef N"),
            "harness #ifndef N dim-default block must be present"
        );
    }

    #[test]
    fn emits_cuda_subject_harness() {
        let op = parse_kernel(GEMV).unwrap();
        let cu = emit_subject_harness(&op, &SubjectSrc::Cuda("my_gemv.cu".into())).unwrap();
        assert!(cu.contains("#include \"my_gemv.cu\""));
        assert!(!cu.contains("jac_run(jac_ctx* ctx) {")); // jac doesn't define run for user subjects
    }

    #[test]
    fn block_table_input_is_seeded_as_a_permutation_not_generic_float_cast() {
        // A minimal paged-kernel shape (BLOCK_TABLE gathering into a page tensor). Uses a
        // `cuda()` subject so this isolates the ALLOC-generation fix under test from any
        // kernel-body matcher/codegen concern (no matcher needs to recognize this toy shape).
        let op = parse_kernel(
            r#"kernel paged_toy {
              tensors {
                Q[NH, HD] : fp16 @global -> stream
                BLOCK_TABLE[MAX_BLOCKS] : int32 @global -> stream
                K_PAGES[NUM_BLOCKS, NKV, BLOCK_SIZE, HD] : fp16 @global -> stream
                O[NH, HD] : fp32 @registers -> global
              }
              params { NH = 4  HD = 8  NKV = 1  MAX_BLOCKS = 4  NUM_BLOCKS = 4  BLOCK_SIZE = 2 }
              phase p over Q.dim(NH) by NH {}
            }"#,
        )
        .unwrap();
        let cu = emit_subject_harness(&op, &SubjectSrc::Cuda("dummy.cu".into())).unwrap();
        // BLOCK_TABLE gets a real Fisher-Yates permutation of [0, NUM_BLOCKS), not the generic
        // per-element float-cast scheme (which would truncate every int32 element to exactly 0).
        assert!(cu.contains("const int nb = NUM_BLOCKS;"));
        assert!(cu.contains("perm[p] = p;"));
        assert!(cu.contains("h[k]=perm[k % (size_t)nb];"));
        // Q (a plain fp16 tensor) is unaffected -- still the original generic seeding.
        assert!(cu.contains("h[k]=(__half)r;"));
        // The generic float-cast-to-int pattern must not appear anywhere: BLOCK_TABLE is the
        // only int32 input in this fixture, so if this string is present the fix didn't apply.
        assert!(!cu.contains("h[k]=(int)r;"));
    }

    #[test]
    fn rejects_multiple_outputs() {
        // A kernel with two `-> global` output tensors must be rejected.
        let op = parse_kernel(
            r#"kernel multi_out {
              tensors { x[N] : fp16 @registers  y[N] : fp32 @registers -> global  z[N] : fp32 @registers -> global }
              params { N = 128 }
              phase p { y = einsum("n -> n", x)  z = einsum("n -> n", x) }
            }"#,
        )
        .unwrap();
        let err = emit_subject_harness(&op, &SubjectSrc::Jac("naive".into()))
            .expect_err("should reject op with >1 output");
        assert!(
            err.contains("multi_out") && err.contains('2'),
            "error message should name the op and output count; got: {err}"
        );
    }
}
