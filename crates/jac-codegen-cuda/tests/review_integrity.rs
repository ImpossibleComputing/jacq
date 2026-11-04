//! Permanent **integrity-invariants** regression suite — `jac-codegen-cuda` slice (F8 silent
//! truncation, F12 degenerate-score guard).
//!
//! F8: the generated runtime read the token file into a fixed `static int prompt[4096]` and never
//! asserted it scored every token — so a 32K "long-context" file was silently truncated to 4096 and
//! still reported as long-context, corrupting reported long-context numbers. The fix sizes the
//! buffer to the model's MAX_SEQ and refuses to truncate; the score runtime additionally rejects
//! <2-token inputs instead of emitting a bogus perplexity of 1.0 (F12).
//!
//! These are pure host-side string-emit checks — `emit_cuda`/`emit_score_cuda` return `String`; no
//! `nvcc`, no GPU. RED on origin/main, GREEN once `overnight/fix-review-gateway-cuda` lands.

use std::path::PathBuf;

/// Lower a committed spec and emit both runtimes (generation + score), exactly as `golden.rs` does.
fn emit(spec: &str) -> (String, String) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../specs/models")
        .join(spec);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let model = jac_syntax::parse_model(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let ir = jac_ir::lower(&model).unwrap_or_else(|e| panic!("{spec}: lower error: {}", e.message));
    (
        jac_codegen_cuda::emit_cuda(&ir),
        jac_codegen_cuda::emit_score_cuda(&ir),
    )
}

#[test]
fn f8_generation_runtime_sizes_prompt_to_max_seq_not_4096() {
    let (cu, _) = emit("tinyllama.jac");
    assert!(
        cu.contains("static int prompt[C::MAX_SEQ]"),
        "the prompt buffer must be sized to the model's MAX_SEQ, not a fixed cap"
    );
    assert!(
        !cu.contains("static int prompt[4096]"),
        "the hard-coded 4096 prompt cap must be gone"
    );
    assert!(
        cu.contains("refusing to truncate"),
        "a token file longer than max_seq must be rejected, not silently truncated"
    );
}

#[test]
fn f8_score_runtime_sizes_prompt_and_refuses_to_truncate() {
    let (_, cu) = emit("tinyllama.jac");
    assert!(
        cu.contains("static int prompt[C::MAX_SEQ]"),
        "the score runtime must size the prompt buffer to MAX_SEQ"
    );
    assert!(
        !cu.contains("static int prompt[4096]"),
        "the score runtime must drop the hard-coded 4096 prompt cap"
    );
    assert!(
        cu.contains("refusing to truncate"),
        "the score runtime must reject over-long token files rather than truncate them"
    );
}

/// F12 (CUDA side) — score mode must reject a <2-token input instead of emitting a bogus perplexity
/// of 1.0 from an empty NLL sum.
#[test]
fn f12_score_runtime_rejects_degenerate_short_input() {
    let (_, cu) = emit("tinyllama.jac");
    assert!(
        cu.contains("score-mode requires at least 2 tokens"),
        "score mode must reject <2-token inputs, not emit a fake perplexity of 1.0"
    );
    assert!(
        !cu.contains("nll_sum / (double)(np - 1) : 0.0"),
        "the np<2 → perplexity-1.0 ternary must be gone"
    );
}
