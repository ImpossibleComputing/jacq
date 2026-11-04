//! The windowed_attn fixture parses and emits a valid cuda-subject harness (ABI for the
//! hand-written masked/skip subjects). No GPU: pure codegen-shape assertions.
use std::path::PathBuf;

fn spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs/models")
}

#[test]
fn fixture_parses_and_emits_cuda_harness() {
    let src = std::fs::read_to_string(spec_dir().join("windowed_attn.jac")).unwrap();
    let op = jac_syntax::parse_kernel(&src).expect("fixture must parse");
    assert_eq!(op.name, "windowed_attn");

    // ABI: 3 inputs (Q,K,V) in decl order, 1 output (O).
    let inputs: Vec<&str> = op
        .tensors
        .iter()
        .filter(|t| !t.output)
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(
        inputs,
        vec!["Q", "K", "V"],
        "input order defines ctx->inputs[0..2]"
    );
    let outs: Vec<&str> = op
        .tensors
        .iter()
        .filter(|t| t.output)
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(outs, vec!["O"], "exactly one -> global output");

    // Emit a cuda-subject harness referencing the (not-yet-written) masked.cu.
    let cu = jac_codegen_cuda::emit_subject_harness(
        &op,
        &jac_syntax::SubjectSrc::Cuda("windowed_attn/masked.cu".into()),
    )
    .expect("harness emit must succeed");

    // -D-overridable dim macros present for every dim.
    for d in ["NH", "HD", "NKV", "L"] {
        assert!(cu.contains(&format!("#ifndef {d}")), "missing #ifndef {d}");
    }
    // The subject file is #included (cuda path).
    assert!(cu.contains("#include \"windowed_attn/masked.cu\""));
    // 3 inputs allocated + 1 output; harness does not define jac_run for cuda subjects.
    assert!(cu.contains("n_in = 3; n_out = 1;"));
    assert!(
        !cu.contains("jac_run(jac_ctx* ctx) {"),
        "cuda subject defines jac_run itself"
    );
}
