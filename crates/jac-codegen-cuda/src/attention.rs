//! Windowed attention codegen (track B, Milestone 1).
//!
//! `emit_windowed_cuda` produces a complete self-contained CUDA runtime whose
//! `attention_score_kernel` applies a sink + recent-window mask before writing
//! each score.  The masking is a no-op when `sink_S = 0` and `window_W = INT_MAX`
//! (the defaults), so the windowed binary matches the exact-attention binary in
//! that configuration.

use jac_ir::ModelIr;

/// The windowed score kernel that replaces the exact `attention_score_kernel`.
/// It has two extra parameters — `sink_S` (number of sink tokens to keep) and
/// `window_W` (recent-window size) — and applies the mask before storing to
/// `scores`.
const WINDOWED_SCORE_KERNEL: &str = r#"// scores[h*MAX_SEQ + s] = masked (Q[h] . K_cache[kv_h, s]) / sqrt(HD).
// Keys outside the sink+recent window get score -INFINITY.
// sink+recent-window mask: attend iff s < sink_S (sink) OR s > pos - window_W (recent).
__global__ void attention_score_kernel(const __half* q, const __half* k_cache,
                                       float* scores, int pos,
                                       int sink_S, int window_W) {
    int h = blockIdx.x;
    int kv_h = h / Config::HEADS_PER_GROUP;
    const __half* qh = q + h * Config::HD;
    const __half* kbase = k_cache + (size_t)kv_h * Config::MAX_SEQ * Config::HD;
    float scale = rsqrtf((float)Config::HD);
    for (int s = threadIdx.x; s <= pos; s += blockDim.x) {
        const __half* ks = kbase + (size_t)s * Config::HD;
        float dot = 0.0f;
        for (int d = 0; d < Config::HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        float sv = dot * scale;
        // Sink+recent-window mask: keep first sink_S tokens + last window_W tokens.
        if (!(s < sink_S || s > pos - window_W)) sv = -INFINITY;
        scores[(size_t)h * Config::MAX_SEQ + s] = sv;
    }
}
"#;

/// The exact `attention_score_kernel` block as it appears in `kernels.cu`
/// (used as the find target for the string replacement).
const EXACT_SCORE_KERNEL: &str = r#"// scores[h*MAX_SEQ + s] = (Q[h] . K_cache[kv_h, s]) / sqrt(HD), for s in [0, pos]. One block
// per query head; GQA maps kv_h = h / HEADS_PER_GROUP.
__global__ void attention_score_kernel(const __half* q, const __half* k_cache,
                                       float* scores, int pos) {
    int h = blockIdx.x;
    int kv_h = h / Config::HEADS_PER_GROUP;
    const __half* qh = q + h * Config::HD;
    const __half* kbase = k_cache + (size_t)kv_h * Config::MAX_SEQ * Config::HD;
    float scale = rsqrtf((float)Config::HD);
    for (int s = threadIdx.x; s <= pos; s += blockDim.x) {
        const __half* ks = kbase + (size_t)s * Config::HD;
        float dot = 0.0f;
        for (int d = 0; d < Config::HD; ++d) dot += __half2float(qh[d]) * __half2float(ks[d]);
        scores[(size_t)h * Config::MAX_SEQ + s] = dot * scale;
    }
}
"#;

/// Build the windowed kernels string by replacing `attention_score_kernel` in the
/// standard kernels template with the windowed version.
///
/// `pub(crate)` so score-mode codegen ([`crate::emit_score_cuda`]) can reuse the
/// same masked kernel — perplexity must flow through the active sink/window policy.
pub(crate) fn windowed_kernels() -> String {
    let kernels = include_str!("templates/kernels.cu");
    assert!(
        kernels.contains(EXACT_SCORE_KERNEL),
        "kernels.cu does not contain the expected exact attention_score_kernel — \
         update EXACT_SCORE_KERNEL in attention.rs to match the template"
    );
    kernels.replace(EXACT_SCORE_KERNEL, WINDOWED_SCORE_KERNEL)
}

/// Apply the windowed-runtime substitutions to a runtime template string.
///
/// The substitutions are:
/// 1. `forward()` signature: add `int sink_S, int window_W` after `bool want_logits`.
/// 2. The `attention_score_kernel` launch: pass `sink_S, window_W`.
/// 3. The prefill `forward(...)` calls: pass `sink_S, window_W`.
/// 4. The generate-loop `forward(...)` call: pass `sink_S, window_W`.
/// 5. The usage line in `main()`: include `[sink_S] [window_W]`.
/// 6. After `int num_new = atoi(argv[3]);`: parse optional sink_S / window_W args.
///
/// `pub(crate)` so score-mode codegen ([`crate::emit_score_cuda`]) can build on the
/// same `forward(... , sink_S, window_W)` plumbing before adding its `--score` branch.
pub(crate) fn apply_windowed_runtime(runtime: &str) -> String {
    runtime
        // 1. forward() declaration signature (20 spaces on continuation line).
        .replace(
            "static void forward(cublasHandle_t handle, const Weights* w, Buffers* b,\n                    int token, int pos, bool want_logits)",
            "static void forward(cublasHandle_t handle, const Weights* w, Buffers* b,\n                    int token, int pos, bool want_logits,\n                    int sink_S, int window_W)",
        )
        // 2. attention_score_kernel launch site.
        .replace(
            "attention_score_kernel<<<C::NH, 128>>>(q, kc, b->scores, pos);",
            "attention_score_kernel<<<C::NH, 128>>>(q, kc, b->scores, pos, sink_S, window_W);",
        )
        // 3a. prefill forward() calls (false = no logits).
        .replace(
            "for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false);",
            "for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false, sink_S, window_W);",
        )
        // 3b. generate-loop forward() call (true = want logits).
        .replace(
            "        forward(handle, &w, &b, cur, pos++, true);",
            "        forward(handle, &w, &b, cur, pos++, true, sink_S, window_W);",
        )
        // 4. usage line (\\n is a C-string escape, so it's literal backslash-n in the file).
        .replace(
            "\"usage: %s <weights_dir> <tokens.bin> <num_new_tokens>\\n\"",
            "\"usage: %s <weights_dir> <tokens.bin> <num_new_tokens> [sink_S] [window_W]\\n\"",
        )
        // 5. arg parsing: insert sink_S / window_W after `num_new` (4-space indent).
        //    Default declarations first so the test can assert on "sink_S = 0" / "INT_MAX".
        .replace(
            "int num_new = atoi(argv[3]);\n    using C = Config;",
            "int num_new = atoi(argv[3]);\n    int sink_S = 0;\n    int window_W = INT_MAX;\n    if (argc > 4) sink_S  = atoi(argv[4]);\n    if (argc > 5) window_W = atoi(argv[5]);\n    if (window_W < 1) { fprintf(stderr, \"error: window_W must be >= 1 (got %d)\\n\", window_W); return 1; }\n    using C = Config;",
        )
}

/// Emit a complete windowed-attention CUDA runtime for `ir`.
///
/// The returned string is a self-contained `.cu` file identical in structure to
/// `emit_cuda` but with the windowed `attention_score_kernel` and the extra
/// `sink_S` / `window_W` plumbing throughout `forward()` and `main()`.
pub fn emit_windowed_cuda(ir: &ModelIr) -> String {
    use jac_ir::Dtype;

    let mut s = String::new();
    s.push_str(&format!(
        "// Generated by jacq — inference runtime for model `{}`.\n\
         // The generator owns this file; regenerate rather than hand-editing.\n\n",
        ir.name
    ));
    s.push_str(super::INCLUDES);
    s.push_str(&super::config_struct(ir));
    s.push_str(&windowed_kernels());

    let base_runtime = match ir.weights.dtype {
        Dtype::Int8 => super::RUNTIME_INT8,
        Dtype::Int4 => super::RUNTIME_INT4,
        _ => super::RUNTIME_FP16,
    };
    s.push_str(&apply_windowed_runtime(base_runtime));
    s
}
