//! Keep the examples a newcomer runs connected to the real parsers and emitters.

use std::path::PathBuf;

fn example(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn tutorial(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/tutorial")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn jac_fences(source: &str) -> Vec<&str> {
    source
        .split("```jac\n")
        .skip(1)
        .map(|rest| {
            assert!(rest.contains("\n```"), "unclosed Jacq fence");
            rest.split("\n```").next().unwrap()
        })
        .collect()
}

fn assert_inner_blocks_in_file(blocks: &str, file: &str) {
    let indented = blocks
        .lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("  {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        file.contains(&indented),
        "tutorial Jacq block drifted from example"
    );
}

#[test]
fn tutorial_jacq_snippets_match_checked_in_sources() {
    let first = tutorial("01-first-compile.md");
    let model_edits = tutorial("02-model-experiments.md");
    let kernel_edit = tutorial("03-kernel-and-memory.md");

    let first_blocks = jac_fences(&first);
    assert_eq!(first_blocks.len(), 1);
    assert_inner_blocks_in_file(first_blocks[0], &example("tinyllama.jac"));

    let model_blocks = jac_fences(&model_edits);
    assert_eq!(model_blocks.len(), 2);
    assert_inner_blocks_in_file(model_blocks[0], &example("tutorial/tinyllama_short.jac"));
    assert_inner_blocks_in_file(model_blocks[1], &example("tutorial/tinyllama_int8.jac"));

    let kernel_blocks = jac_fences(&kernel_edit);
    assert_eq!(kernel_blocks.len(), 2);
    assert_inner_blocks_in_file(kernel_blocks[0], &example("gemv.jac"));
    assert_eq!(
        kernel_blocks[1],
        example("tutorial/gemv_small.jac").trim_end()
    );
    jac_syntax::parse_kernel(kernel_blocks[1]).expect("inline edited GEMV must parse");
}

#[test]
fn tinyllama_model_emits_runtime_and_exporter() {
    let model = jac_syntax::parse_model(&example("tinyllama.jac")).unwrap();
    let ir = jac_ir::lower(&model).unwrap();
    assert_eq!(ir.name, "tinyllama");
    let emitted = jac_codegen_cuda::emit(&ir).unwrap();
    assert!(emitted.cuda.contains("struct Config"));
    assert!(emitted
        .python
        .contains("TinyLlama/TinyLlama-1.1B-Chat-v1.0"));
}

#[test]
fn gemv_kernel_emits_standalone_cuda() {
    let kernel = jac_syntax::parse_kernel(&example("gemv.jac")).unwrap();
    let cuda =
        jac_codegen_cuda::emit_kernel(&kernel, jac_codegen_cuda::KernelVariant::Naive).unwrap();
    assert!(cuda.contains("int main("));
    assert!(cuda.contains("gemv_kernel<<<"));
    assert!(cuda.contains("static constexpr int M = 4096"));
    assert!(cuda.contains("int block = 256"));
}

#[test]
fn tutorial_gemv_edit_changes_only_name_and_output_rows() {
    let base = jac_syntax::parse_kernel(&example("gemv.jac")).unwrap();
    let edited = jac_syntax::parse_kernel(&example("tutorial/gemv_small.jac")).unwrap();
    let mut expected = base;
    expected.name = "gemv_small".into();
    expected
        .params
        .iter_mut()
        .find(|(key, _)| key == "M")
        .unwrap()
        .1 = 512;
    assert_eq!(edited, expected);

    let cuda =
        jac_codegen_cuda::emit_kernel(&edited, jac_codegen_cuda::KernelVariant::Naive).unwrap();
    assert!(cuda.contains("static constexpr int M = 512;"));
    assert!(cuda.contains("static constexpr int N = 128;"));
    assert!(cuda.contains("int block = 256;"));
    assert!(cuda.contains("gemv_small_kernel<<<"));
}

#[test]
fn tutorial_variants_change_only_the_taught_codegen_inputs() {
    use jac_syntax::ast::Value;

    let base = jac_ir::lower(&jac_syntax::parse_model(&example("tinyllama.jac")).unwrap()).unwrap();
    let short =
        jac_ir::lower(&jac_syntax::parse_model(&example("tutorial/tinyllama_short.jac")).unwrap())
            .unwrap();
    let int8 =
        jac_ir::lower(&jac_syntax::parse_model(&example("tutorial/tinyllama_int8.jac")).unwrap())
            .unwrap();

    let mut expected_short_config = base.config.clone();
    expected_short_config.max_seq = 256;
    assert_eq!(short.config, expected_short_config);
    assert_eq!(short.weights, base.weights);

    let mut expected_int8_weights = base.weights.clone();
    expected_int8_weights.dtype = jac_ir::Dtype::Int8;
    assert_eq!(int8.config, base.config);
    assert_eq!(int8.weights, expected_int8_weights);

    // AST equality also checks layer statements, pipeline, source and format literals,
    // and every unchanged config entry, not just the selected lowered fields.
    let mut expected_short_model = base.model.clone();
    expected_short_model.name = "tinyllama_short".into();
    expected_short_model
        .config
        .iter_mut()
        .find(|(key, _)| key == "max_seq")
        .unwrap()
        .1 = Value::Int(256);
    assert_eq!(short.model, expected_short_model);

    let mut expected_int8_model = base.model.clone();
    expected_int8_model.name = "tinyllama_int8".into();
    expected_int8_model
        .weights
        .iter_mut()
        .find(|(key, _)| key == "dtype")
        .unwrap()
        .1 = Value::Ident("int8".into());
    assert_eq!(int8.model, expected_int8_model);

    let base_out = jac_codegen_cuda::emit(&base).unwrap();
    let short_out = jac_codegen_cuda::emit(&short).unwrap();
    let int8_out = jac_codegen_cuda::emit(&int8).unwrap();
    assert!(base_out.cuda.contains("MAX_SEQ         = 2048"));
    assert!(short_out.cuda.contains("MAX_SEQ         = 256"));
    assert!(int8_out.cuda.contains("MAX_SEQ         = 2048"));
    assert!(int8_out.python.contains("DTYPE = \"int8\""));
    assert!(int8_out.cuda.contains("signed char"));
}
