//! Golden-snapshot tests for the generated CUDA runtimes.
//!
//! Each committed `specs/*.jac` is lowered and code-generated, then compared byte-for-byte
//! against a reviewed reference under `tests/snapshots/`. These lock in the GPU-validated output
//! (fp16/int8/int4 TinyLlama + Qwen2) so a refactor can't silently change a kernel or config
//! constant. Reading the real specs also catches drift between a spec and the code generator.
//!
//! After an intentional codegen change, regenerate and review the diffs:
//!   UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden

use std::path::PathBuf;

fn specs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs/models")
}

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

/// Emit CUDA for `spec` and assert it matches `tests/snapshots/<snapshot>` (or rewrite it when
/// `UPDATE_SNAPSHOTS` is set).
fn check(spec: &str, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let model = jac_syntax::parse_model(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let ir = jac_ir::lower(&model).unwrap_or_else(|e| panic!("{spec}: lower error: {}", e.message));
    let cuda = jac_codegen_cuda::emit_cuda(&ir);

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden`",
            snap_path.display()
        )
    });
    if cuda != expected {
        panic!("{}", first_diff(spec, snapshot, &cuda, &expected));
    }
}

/// Emit the Python exporter for `spec` and assert it matches `tests/snapshots/<snapshot>`
/// (or rewrite it when `UPDATE_SNAPSHOTS` is set). Mirrors `check()` for the `.py` side.
fn check_py(spec: &str, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let model = jac_syntax::parse_model(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let ir = jac_ir::lower(&model).unwrap_or_else(|e| panic!("{spec}: lower error: {}", e.message));
    let emitted = jac_codegen_cuda::emit(&ir).unwrap_or_else(|e| panic!("{spec}: emit error: {e}"));

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &emitted.python).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden`",
            snap_path.display()
        )
    });
    if emitted.python != expected {
        panic!("{}", first_diff(spec, snapshot, &emitted.python, &expected));
    }
}

/// Build a compact failure message pointing at the first differing line.
fn first_diff(spec: &str, snapshot: &str, got: &str, expected: &str) -> String {
    let g: Vec<&str> = got.lines().collect();
    let e: Vec<&str> = expected.lines().collect();
    let mut i = 0;
    while i < g.len().min(e.len()) && g[i] == e[i] {
        i += 1;
    }
    let gl = g.get(i).copied().unwrap_or("<end of file>");
    let el = e.get(i).copied().unwrap_or("<end of file>");
    format!(
        "generated CUDA for `{spec}` differs from snapshot `{snapshot}` at line {}:\n  \
         expected: {el}\n  got:      {gl}\n(line counts: got {} vs expected {}).\n\
         If this change is intentional, regenerate and review with \
         `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden`.",
        i + 1,
        g.len(),
        e.len()
    )
}

#[test]
fn snapshot_tinyllama_fp16() {
    check("tinyllama.jac", "tinyllama.cu");
}

#[test]
fn snapshot_tinyllama_int8() {
    check("tinyllama_int8.jac", "tinyllama_int8.cu");
}

#[test]
fn snapshot_tinyllama_int4() {
    check("tinyllama_int4.jac", "tinyllama_int4.cu");
}

#[test]
fn snapshot_qwen2_0_5b() {
    check("qwen2_0_5b.jac", "qwen2_0_5b.cu");
}

#[test]
fn snapshot_tinyllama_lora_wo() {
    check("tinyllama_lora_wo.jac", "tinyllama_lora_wo.cu");
}

#[test]
fn snapshot_tinyllama_lora_qkvo() {
    check("tinyllama_lora_qkvo.jac", "tinyllama_lora_qkvo.cu");
}

#[test]
fn snapshot_qwen2_int8() {
    check("qwen2_0_5b_int8.jac", "qwen2_0_5b_int8.cu");
}

#[test]
fn snapshot_qwen2_int4() {
    check("qwen2_0_5b_int4.jac", "qwen2_0_5b_int4.cu");
}

#[test]
fn snapshot_tinyllama_lora_qkvo_int8() {
    check(
        "tinyllama_lora_qkvo_int8.jac",
        "tinyllama_lora_qkvo_int8.cu",
    );
}

#[test]
fn snapshot_tinyllama_lora_qkvo_int4() {
    check(
        "tinyllama_lora_qkvo_int4.jac",
        "tinyllama_lora_qkvo_int4.cu",
    );
}

// ── Python exporter golden snapshots ─────────────────────────────────────────
//
// These lock in the generated weight-export script byte-for-byte, covering the
// same four representative model configurations as the CUDA runtime snapshots
// above. The `.py` output is pure template substitution (source repo, layer
// count, dtype, attn_bias, LoRA flags), so even small template edits are
// visible here without having to diff the template file itself.
//
// Regenerate after an intentional change:
//   UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden

#[test]
fn snapshot_py_tinyllama_fp16() {
    check_py("tinyllama.jac", "tinyllama.py");
}

#[test]
fn snapshot_py_tinyllama_int8() {
    check_py("tinyllama_int8.jac", "tinyllama_int8.py");
}

#[test]
fn snapshot_py_tinyllama_int4() {
    check_py("tinyllama_int4.jac", "tinyllama_int4.py");
}

#[test]
fn snapshot_py_qwen2_0_5b() {
    check_py("qwen2_0_5b.jac", "qwen2_0_5b.py");
}

#[test]
fn snapshot_py_tinyllama_lora_wo() {
    check_py("tinyllama_lora_wo.jac", "tinyllama_lora_wo.py");
}

// ── windowed-attention golden (track B M1) ──────────────────────────────────

/// Mirror of `check` but calls `emit_windowed_cuda`.
fn check_windowed(spec: &str, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let model = jac_syntax::parse_model(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let ir = jac_ir::lower(&model).unwrap_or_else(|e| panic!("{spec}: lower error: {}", e.message));
    let cuda = jac_codegen_cuda::emit_windowed_cuda(&ir);

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden`",
            snap_path.display()
        )
    });
    if cuda != expected {
        panic!("{}", first_diff(spec, snapshot, &cuda, &expected));
    }
}

/// (b) Windowed golden: emit_windowed_cuda with a finite window contains the masking code.
#[test]
fn windowed_attention_contains_mask_code() {
    let src = r#"
    model tinyllama {
      config { hidden = 2048  layers = 22  heads = 32  kv_heads = 4
               head_dim = 64  ffn_dim = 5632  vocab = 32000
               max_seq = 2048  rope_base = 10000.0  norm_eps = 1e-5 }
      weights { format = huggingface  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  dtype = fp16 }
      layer decoder { x = rmsnorm(input, w, norm_eps) }
      pipeline { embed(t, e) -> greedy_sample }
    }"#;
    let ir = jac_ir::lower(&jac_syntax::parse_model(src).unwrap()).unwrap();
    let cu = jac_codegen_cuda::emit_windowed_cuda(&ir);
    // Must contain the windowed score kernel params.
    assert!(
        cu.contains("sink_S"),
        "windowed kernel missing sink_S param"
    );
    assert!(
        cu.contains("window_W"),
        "windowed kernel missing window_W param"
    );
    // Must contain the -INFINITY masking.
    assert!(
        cu.contains("-INFINITY") || cu.contains("INFINITY"),
        "windowed kernel missing -INFINITY mask"
    );
    // Must contain the mask condition.
    assert!(
        cu.contains("pos - window_W"),
        "windowed kernel missing pos-window_W condition"
    );
    // Must contain the main() usage line with sink/window args.
    assert!(
        cu.contains("sink_S") && cu.contains("window_W"),
        "main() missing sink/window args"
    );
    // Snapshot this golden.
    check_windowed("tinyllama.jac", "tinyllama_windowed.cu");
}

/// (b') Windowed golden for Qwen2: the windowed path is architecture-agnostic, so
/// `emit_windowed_cuda` must apply the same sink/window masking to Qwen2 (GQA 14/2,
/// rope_base 1e6, attn_bias) as to TinyLlama. This proves windowing composes with the
/// attn-bias path — the masked `attention_score_kernel` and the `forward(..., sink_S,
/// window_W)` plumbing sit *upstream* of the bias-add, which is unchanged.
#[test]
fn windowed_attention_qwen2_contains_mask_code() {
    let src = r#"
    model qwen2 {
      config { hidden = 896  layers = 24  heads = 14  kv_heads = 2
               head_dim = 64  ffn_dim = 4864  vocab = 151936
               max_seq = 2048  rope_base = 1000000.0  norm_eps = 1e-6  attn_bias = true }
      weights { format = huggingface  source = "Qwen/Qwen2-0.5B-Instruct"  dtype = fp16 }
      layer decoder { x = rmsnorm(input, w, norm_eps) }
      pipeline { embed(t, e) -> greedy_sample }
    }"#;
    let ir = jac_ir::lower(&jac_syntax::parse_model(src).unwrap()).unwrap();
    let cu = jac_codegen_cuda::emit_windowed_cuda(&ir);
    // Same windowed score-kernel params + mask as TinyLlama (architecture-agnostic path).
    assert!(
        cu.contains("sink_S"),
        "windowed kernel missing sink_S param"
    );
    assert!(
        cu.contains("window_W"),
        "windowed kernel missing window_W param"
    );
    assert!(
        cu.contains("-INFINITY") || cu.contains("INFINITY"),
        "windowed kernel missing -INFINITY mask"
    );
    assert!(
        cu.contains("pos - window_W"),
        "windowed kernel missing pos-window_W condition"
    );
    // Qwen2-specific: the attn-bias path is preserved alongside the windowing.
    assert!(
        cu.contains("static constexpr bool  ATTN_BIAS       = true;"),
        "qwen2 ATTN_BIAS flag missing"
    );
    assert!(
        cu.contains("bias_add_kernel"),
        "qwen2 bias-add kernel missing under windowing"
    );
    // Qwen2 config constants flow through unchanged (GQA 14/2, rope_base 1e6).
    assert!(
        cu.contains("static constexpr int   NH              = 14;"),
        "qwen2 head count missing"
    );
    assert!(
        cu.contains("static constexpr int   NKV             = 2;"),
        "qwen2 kv-head count missing"
    );
    assert!(
        cu.contains("ROPE_BASE       = 1000000.0f;"),
        "qwen2 rope_base missing"
    );
    // Snapshot this golden (mirrors how score-mode snapshots qwen2_0_5b_score.cu).
    check_windowed("qwen2_0_5b.jac", "qwen2_0_5b_windowed.cu");
}

// ── teacher-forced score-mode golden (track B, perplexity) ──────────────────

/// Mirror of `check` but calls `emit_score_cuda`.
fn check_score(spec: &str, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let model = jac_syntax::parse_model(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let ir = jac_ir::lower(&model).unwrap_or_else(|e| panic!("{spec}: lower error: {}", e.message));
    let cuda = jac_codegen_cuda::emit_score_cuda(&ir);

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden`",
            snap_path.display()
        )
    });
    if cuda != expected {
        panic!("{}", first_diff(spec, snapshot, &cuda, &expected));
    }
}

/// Score-mode golden: the teacher-forced runtime carries the NLL kernel, the `--score`
/// branch, and the windowed mask it composes with. Snapshots both real architectures
/// (TinyLlama fp16 and Qwen2 fp16, the latter exercising the attn-bias path).
#[test]
fn score_mode_contains_nll_and_perplexity() {
    let src = r#"
    model tinyllama {
      config { hidden = 2048  layers = 22  heads = 32  kv_heads = 4
               head_dim = 64  ffn_dim = 5632  vocab = 32000
               max_seq = 2048  rope_base = 10000.0  norm_eps = 1e-5 }
      weights { format = huggingface  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  dtype = fp16 }
      layer decoder { x = rmsnorm(input, w, norm_eps) }
      pipeline { embed(t, e) -> greedy_sample }
    }"#;
    let ir = jac_ir::lower(&jac_syntax::parse_model(src).unwrap()).unwrap();
    let cu = jac_codegen_cuda::emit_score_cuda(&ir);
    // Score kernel + teacher-forced launch over the known next token.
    assert!(
        cu.contains("__global__ void nll_kernel("),
        "score runtime missing nll_kernel"
    );
    assert!(
        cu.contains("nll_kernel<<<1, 256>>>(d_nll, b.logits, prompt[t + 1], C::VOCAB);"),
        "score runtime missing teacher-forced nll launch"
    );
    // f32 scores.bin output + the perplexity line.
    assert!(
        cu.contains(r#"fopen("scores.bin", "wb")"#),
        "score runtime missing scores.bin"
    );
    assert!(
        cu.contains("perplexity:"),
        "score runtime missing perplexity line"
    );
    // Composes with the knob: the windowed mask is present and threaded into the score loop.
    assert!(
        cu.contains("int sink_S, int window_W)"),
        "score runtime missing windowed forward()"
    );
    assert!(
        cu.contains("pos - window_W"),
        "score runtime missing sink/window mask"
    );
    // Snapshot both real fp16 architectures.
    check_score("tinyllama.jac", "tinyllama_score.cu");
    check_score("qwen2_0_5b.jac", "qwen2_0_5b_score.cu");
}

/// (c) Behavioral no-op: with W=INT_MAX and S=0, the default args in main() are 0 and INT_MAX.
#[test]
fn windowed_attention_default_args_are_exact() {
    let src = r#"
    model tinyllama {
      config { hidden = 2048  layers = 22  heads = 32  kv_heads = 4
               head_dim = 64  ffn_dim = 5632  vocab = 32000
               max_seq = 2048  rope_base = 10000.0  norm_eps = 1e-5 }
      weights { format = huggingface  source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  dtype = fp16 }
      layer decoder { x = rmsnorm(input, w, norm_eps) }
      pipeline { embed(t, e) -> greedy_sample }
    }"#;
    let ir = jac_ir::lower(&jac_syntax::parse_model(src).unwrap()).unwrap();
    let cu = jac_codegen_cuda::emit_windowed_cuda(&ir);
    // Default sink_S = 0.
    assert!(
        cu.contains("sink_S = 0"),
        "default sink_S should be 0 (exact)"
    );
    // Default window_W = INT_MAX.
    assert!(
        cu.contains("INT_MAX"),
        "default window_W should be INT_MAX (exact)"
    );
}
