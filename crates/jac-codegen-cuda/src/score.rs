//! Teacher-forced score-mode codegen (track B, perplexity).
//!
//! `emit_score_cuda` produces a complete self-contained CUDA runtime that adds a
//! `--score` path to the generated `main()`: given a token file, it runs the forward
//! teacher-forced over the sequence, accumulates the per-token negative log-likelihood
//! of the *known* next token (not the argmax), writes `scores.bin` (f32, length N-1),
//! and prints `perplexity: <exp(mean NLL)>`.
//!
//! It is built on the *windowed* runtime so scoring composes with the active
//! `--sink S / --window W` SelectionPolicy — perplexity flows through the masked
//! attention exactly as a real long-context decode would. At `(sink=0, window=INT_MAX)`
//! the mask is a no-op, so score-mode reproduces exact-attention scoring (the
//! correctness anchor against a torch teacher-forced reference).
//!
//! Generation behavior is unchanged: the `--score` branch is additive and only taken
//! when the num-tokens slot is the `--score` flag. fp16 only; int8/int4 mirror later.

use jac_ir::ModelIr;

/// Per-token negative-log-likelihood kernel: `out_nll[0] = -log_softmax(logits)[target]`.
///
/// One block; a two-pass warp-shuffle reduction over the vocab (running max for
/// numerical stability, then sum of `exp(logit - max)`), mirroring `softmax_kernel`.
/// `-log_softmax(target) = logsumexp(logits) - logits[target] = (max + log sum) - logits[target]`.
const SCORE_KERNEL: &str = r#"
// ---- score-mode kernel (teacher-forced per-token NLL) ----

// out_nll[0] = -log_softmax(logits)[target] = logsumexp(logits) - logits[target].
// One block, warp-shuffle reduction over the vocab (mirrors softmax_kernel's two passes).
__global__ void nll_kernel(float* out_nll, const __half* logits, int target, int n) {
    int tid = threadIdx.x;
    __shared__ float red[32];

    // Pass 1: running max for numerical stability.
    float m = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) m = fmaxf(m, __half2float(logits[i]));
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float mm = red[0];
        for (int i = 1; i < nw; ++i) mm = fmaxf(mm, red[i]);
        red[0] = mm;
    }
    __syncthreads();
    float mx = red[0];

    // Pass 2: sum of exp(logit - max), then the log-softmax NLL of the target token.
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += expf(__half2float(logits[i]) - mx);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float ss = 0.0f;
        for (int i = 0; i < nw; ++i) ss += red[i];
        if (ss <= 0.0f) ss = 1.0e-30f;  // guard: all-masked window makes denom zero → avoid log(0)
        *out_nll = (mx + logf(ss)) - __half2float(logits[target]);
    }
}
"#;

/// Apply the score-mode substitutions to a *windowed* runtime string.
///
/// Operates on the output of [`crate::attention::apply_windowed_runtime`], so the
/// `forward(..., sink_S, window_W)` plumbing is already in place. The substitutions:
/// 1. usage line: document the `--score` alternative to `<num_new_tokens>`.
/// 2. arg parse: detect `--score` in the num-tokens slot (`argv[3]`).
/// 3. `main()`: insert the teacher-forced scoring branch before the generation prefill.
fn apply_score_runtime(runtime: &str) -> String {
    let scored = runtime
        // 1. usage line — add the `--score` alternative (leave the `\n` escape untouched).
        .replace(
            "<num_new_tokens> [sink_S] [window_W]",
            "<num_new_tokens|--score> [sink_S] [window_W]",
        )
        // 2. detect score-mode from the num-tokens slot. num_new is never negative in
        //    generation, so a leading '-' (i.e. "--score") unambiguously selects scoring.
        .replace(
            "    int num_new = atoi(argv[3]);\n    int sink_S = 0;",
            "    // \"--score\" in the num-tokens slot selects teacher-forced perplexity over <tokens.bin>.\n    \
             bool score_mode = (argv[3][0] == '-');\n    \
             int num_new = score_mode ? 0 : atoi(argv[3]);\n    int sink_S = 0;",
        )
        // 3. insert the score-mode branch right after `int pos = 0;`, before the prefill.
        .replace(SCORE_PREFILL_ANCHOR, SCORE_BRANCH);
    assert!(
        scored.contains("score_mode")
            && scored.contains("nll_kernel<<<")
            && scored.contains("perplexity:"),
        "score-mode substitutions did not all apply — the windowed runtime template changed; \
         update apply_score_runtime in score.rs to match"
    );
    scored
}

/// The windowed-runtime prefill region the score branch is injected ahead of.
/// (No `\` line-continuation here: the leading 4-space indent of the first line is
/// part of the anchor, so the match leaves no stray indentation behind it.)
const SCORE_PREFILL_ANCHOR: &str = "    // Prefill all but the last prompt token (fills KV cache; no sampling).
    int pos = 0;
    for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false, sink_S, window_W);";

/// The teacher-forced scoring branch + the (now comment-free) prefill that follows it.
const SCORE_BRANCH: &str = r#"    int pos = 0;
    if (score_mode) {
        // Teacher-forced scoring: feed each known token, accumulate the NLL of the
        // actual next token, and append the GIVEN token to the KV cache (not the argmax).
        // Composes with the sink/window policy above; at (0, INT_MAX) this is exact.
        if (np < 2) {
            fprintf(stderr, "error: score-mode requires at least 2 tokens (got %d) — cannot compute NLL\n", np);
            cublasDestroy(handle); return 1;
        }
        float* d_nll;
        CUDA_CHECK(cudaMalloc(&d_nll, sizeof(float)));
        float* nll_host = (float*)malloc((size_t)(np - 1) * sizeof(float));
        double nll_sum = 0.0;
        for (int t = 0; t < np - 1; ++t) {
            forward(handle, &w, &b, prompt[t], pos++, true, sink_S, window_W);
            nll_kernel<<<1, 256>>>(d_nll, b.logits, prompt[t + 1], C::VOCAB);
            float nll = 0.0f;
            CUDA_CHECK(cudaMemcpy(&nll, d_nll, sizeof(float), cudaMemcpyDeviceToHost));
            nll_host[t] = nll;
            nll_sum += nll;
        }
        FILE* sf = fopen("scores.bin", "wb");
        if (!sf) { fprintf(stderr, "cannot write scores.bin\n"); return 1; }
        fwrite(nll_host, sizeof(float), (size_t)(np - 1), sf);
        fclose(sf);
        double mean_nll = nll_sum / (double)(np - 1);
        printf("perplexity: %.6f\n", exp(mean_nll));
        free(nll_host);
        cudaFree(d_nll);
        cublasDestroy(handle);
        return 0;
    }

    // Prefill all but the last prompt token (fills KV cache; no sampling).
    for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false, sink_S, window_W);"#;

/// Emit a complete teacher-forced score-mode CUDA runtime for `ir` (fp16).
///
/// Structure mirrors [`crate::emit_windowed_cuda`]: header + includes + config + the
/// windowed kernels + the score (`nll_kernel`) kernel + the windowed runtime with the
/// `--score` branch spliced into `main()`. fp16 only — the score path reads the fp16
/// `b.logits` the generation loop already computes.
pub fn emit_score_cuda(ir: &ModelIr) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "// Generated by jacq — inference runtime for model `{}`.\n\
         // The generator owns this file; regenerate rather than hand-editing.\n\n",
        ir.name
    ));
    s.push_str(super::INCLUDES);
    s.push_str(&super::config_struct(ir));
    s.push_str(&super::attention::windowed_kernels());
    s.push_str(SCORE_KERNEL);
    s.push_str(&apply_score_runtime(
        &super::attention::apply_windowed_runtime(super::RUNTIME_FP16),
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use jac_syntax::parse_model;

    fn tiny_ir() -> ModelIr {
        let src = r#"
        model tinyllama {
          config { hidden = 2048  layers = 22  heads = 32  kv_heads = 4
                   head_dim = 64  ffn_dim = 5632  vocab = 32000
                   max_seq = 2048  rope_base = 10000.0  norm_eps = 1e-5 }
          weights { format = huggingface  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  dtype = fp16 }
          layer decoder { x = rmsnorm(input, w, norm_eps) }
          pipeline { embed(t, e) -> greedy_sample }
        }"#;
        jac_ir::lower(&parse_model(src).unwrap()).unwrap()
    }

    #[test]
    fn score_runtime_has_nll_kernel_and_branch() {
        let cu = emit_score_cuda(&tiny_ir());
        // The score kernel and its teacher-forced launch.
        assert!(
            cu.contains("__global__ void nll_kernel("),
            "missing nll_kernel definition"
        );
        assert!(
            cu.contains("nll_kernel<<<1, 256>>>(d_nll, b.logits, prompt[t + 1], C::VOCAB);"),
            "missing teacher-forced nll_kernel launch over the known next token"
        );
        // The CLI branch, the f32 output, and the perplexity line.
        assert!(
            cu.contains("bool score_mode = (argv[3][0] == '-');"),
            "missing --score detection"
        );
        assert!(
            cu.contains(r#"fopen("scores.bin", "wb")"#),
            "missing scores.bin output"
        );
        assert!(
            cu.contains(r#"printf("perplexity: %.6f\n", exp(mean_nll));"#),
            "missing perplexity line"
        );
        assert!(
            cu.contains("<num_new_tokens|--score>"),
            "usage line not updated for --score"
        );
    }

    #[test]
    fn score_runtime_rejects_single_token() {
        // np<2 must produce an error message and non-zero exit, not perplexity 1.0.
        let cu = emit_score_cuda(&tiny_ir());
        assert!(
            cu.contains("score-mode requires at least 2 tokens"),
            "missing np<2 diagnostic"
        );
        assert!(
            !cu.contains("nll_sum / (double)(np - 1) : 0.0"),
            "old np>1 ternary still present — not guarded"
        );
    }

    #[test]
    fn score_runtime_guards_softmax_denom() {
        // Softmax denominator must be clamped to prevent log(0) when window mask zeroes all logits.
        let cu = emit_score_cuda(&tiny_ir());
        assert!(
            cu.contains("if (ss <= 0.0f) ss = 1.0e-30f;"),
            "missing softmax denom guard in nll_kernel"
        );
    }

    #[test]
    fn score_runtime_uses_max_seq_for_prompt_buffer() {
        // F8: prompt buffer must use C::MAX_SEQ, not the hard-coded 4096 sentinel.
        let cu = emit_score_cuda(&tiny_ir());
        assert!(
            cu.contains("static int prompt[C::MAX_SEQ]"),
            "prompt buffer must be sized to C::MAX_SEQ"
        );
        assert!(
            !cu.contains("static int prompt[4096]"),
            "old 4096 hard-coded buffer must not appear"
        );
        assert!(
            cu.contains("refusing to truncate"),
            "missing overlong-file rejection message"
        );
    }

    #[test]
    fn windowed_runtime_rejects_window_w_zero() {
        // F12: window_W < 1 must produce an error (all-masked attention causes NaN softmax).
        let cu = emit_score_cuda(&tiny_ir());
        assert!(
            cu.contains("window_W must be >= 1"),
            "missing window_W<1 rejection"
        );
    }

    #[test]
    fn score_runtime_composes_with_window() {
        // Built on the windowed base: the masked kernel + sink/window plumbing must be present,
        // so a (S,W) sweep can collect perplexity through windowed attention.
        let cu = emit_score_cuda(&tiny_ir());
        assert!(
            cu.contains("int sink_S, int window_W)"),
            "missing windowed forward() signature"
        );
        assert!(
            cu.contains("if (!(s < sink_S || s > pos - window_W)) sv = -INFINITY;"),
            "missing sink+recent-window mask"
        );
        assert!(
            cu.contains("forward(handle, &w, &b, prompt[t], pos++, true, sink_S, window_W);"),
            "score loop must thread sink_S/window_W into forward()"
        );
    }

    #[test]
    fn score_does_not_disturb_generation() {
        // The generation prefill + decode loop survive unchanged alongside the score branch.
        let cu = emit_score_cuda(&tiny_ir());
        assert!(cu.contains("for (int i = 0; i < np - 1; ++i) forward(handle, &w, &b, prompt[i], pos++, false, sink_S, window_W);"),
                "generation prefill missing");
        assert!(
            cu.contains("argmax_kernel<<<1, 256>>>(b.d_token, b.logits, C::VOCAB);"),
            "generation decode loop missing"
        );
    }
}
