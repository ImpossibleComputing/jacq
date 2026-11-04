//! Semantic analysis and lowering for the Jacquard model DSL.
//!
//! Takes a parsed [`jac_syntax::ast::Model`], applies config defaults, validates required
//! fields, and computes the derived quantities the CUDA backend needs (`kv_dim`, `qkv_out`,
//! `heads_per_group`, attention mode). The result is a [`ModelIr`] the code generator consumes.
//! Also lowers `experiment` specs into [`ExperimentIr`] (validation + sweep expansion).

use jac_syntax::ast::{LoraSpec, Model, Value};
use jac_syntax::{ExpKind, Experiment, Subject};

/// A semantic-analysis failure (missing field, bad value, inconsistent dims).
#[derive(Debug, Clone, PartialEq)]
pub struct SemaError {
    pub message: String,
}

impl SemaError {
    fn new(msg: impl Into<String>) -> SemaError {
        SemaError {
            message: msg.into(),
        }
    }
}

/// How query heads map to KV heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    /// `kv_heads == heads`
    Mha,
    /// `1 < kv_heads < heads`
    Gqa,
    /// `kv_heads == 1`
    Mqa,
}

/// Weight source format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    HuggingFace,
    Gguf,
}

/// Weight data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp16,
    Int8,
    Int4,
    Q4k,
    W3,
    GgufQ4km,
}

/// Config with defaults applied and derived dimensions computed.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    // declared / defaulted
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub ffn_dim: i64,
    pub vocab: i64,
    pub max_seq: i64,
    pub rope_base: f64,
    pub norm_eps: f64,
    pub kv_dtype: Option<Dtype>,
    /// Whether the q/k/v projections carry a bias (true for Qwen2; false for LLaMA-family).
    pub attn_bias: bool,
    // derived
    pub q_dim: i64,
    pub kv_dim: i64,
    pub qkv_out: i64,
    pub heads_per_group: i64,
    pub attention: Attention,
}

/// Resolved weights block.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub format: WeightFormat,
    pub source: String,
    pub dtype: Dtype,
}

/// The lowered, validated model ready for code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelIr {
    pub name: String,
    pub config: Config,
    pub weights: Weights,
    /// Validated LoRA adapter spec, if the model declares a `lora` block.
    pub lora: Option<LoraSpec>,
    pub model: Model,
}

/// A single point in the sweep: resolved dim values, a stable id, and a deterministic seed.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    pub id: String,
    pub seed: u64,
    pub defines: Vec<(String, i64)>,
}

/// A validated experiment ready for the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentIr {
    pub name: String,
    pub op_path: String,
    pub subjects: Vec<Subject>,
    pub reference: String,
    pub scenarios: Vec<Scenario>,
    pub metrics: Vec<String>,
}

/// Validate an experiment and expand its sweep into the scenario list.
pub fn lower_experiment(e: &Experiment) -> Result<ExperimentIr, SemaError> {
    if e.kind != ExpKind::Kernel {
        return Err(SemaError::new(
            "only `kind = kernel` experiments are supported",
        ));
    }
    if e.subjects.is_empty() {
        return Err(SemaError::new("experiment has no subjects"));
    }
    if !e.subjects.iter().any(|s| s.name == e.reference) {
        return Err(SemaError::new(format!(
            "reference `{}` is not one of the subjects",
            e.reference
        )));
    }
    let op_path = match e.fixture.iter().find(|(k, _)| k == "op") {
        Some((_, Value::Str(s))) => s.clone(),
        _ => {
            return Err(SemaError::new(
                "kernel experiment fixture needs `op = \"<kernel.jac>\"`",
            ))
        }
    };
    // Cartesian product of sweep axes (row-major over axis order), stable ids + seeds.
    // Start with one empty scenario; an experiment with no sweep axes therefore runs once
    // with no -D overrides. Each axis multiplies the scenario set (cartesian product).
    let mut scenarios = vec![Scenario {
        id: String::new(),
        seed: 0,
        defines: vec![],
    }];
    for axis in &e.sweep {
        let mut next = Vec::new();
        for base in &scenarios {
            for &v in &axis.values {
                let mut defines = base.defines.clone();
                defines.push((axis.name.clone(), v));
                next.push(Scenario {
                    id: String::new(),
                    seed: 0,
                    defines,
                });
            }
        }
        scenarios = next;
    }
    for (i, s) in scenarios.iter_mut().enumerate() {
        // Phase A: axis names are identifiers and values are non-negative literals, so the
        // concatenated id (e.g. "M1024_N128") is unambiguous and stable.
        s.id = s
            .defines
            .iter()
            .map(|(k, v)| format!("{k}{v}"))
            .collect::<Vec<_>>()
            .join("_");
        s.seed = (i as u64) + 1;
    }
    Ok(ExperimentIr {
        name: e.name.clone(),
        op_path,
        subjects: e.subjects.clone(),
        reference: e.reference.clone(),
        scenarios,
        metrics: e.metrics.iter().map(|m| m.name.clone()).collect(),
    })
}

/// Base linears a LoRA adapter may target.
const LORA_TARGETS: [&str; 4] = ["w_q", "w_k", "w_v", "w_o"];

/// Lower and validate a parsed model.
pub fn lower(model: &Model) -> Result<ModelIr, SemaError> {
    let config = resolve_config(model)?;
    let weights = resolve_weights(model)?;
    if model.layers.is_empty() {
        return Err(SemaError::new("model has no `layer` definition"));
    }
    if model.pipeline.is_empty() {
        return Err(SemaError::new("model has no `pipeline`"));
    }
    let lora = match &model.lora {
        Some(l) => Some(validate_lora(l)?),
        None => None,
    };
    Ok(ModelIr {
        name: model.name.clone(),
        config,
        weights,
        lora,
        model: model.clone(),
    })
}

/// Validate a LoRA spec: positive rank/alpha and a non-empty target set drawn from the base
/// linears (`w_q`, `w_k`, `w_v`, `w_o`).
fn validate_lora(l: &LoraSpec) -> Result<LoraSpec, SemaError> {
    if l.rank <= 0 {
        return Err(SemaError::new("lora `rank` must be positive"));
    }
    if l.alpha <= 0.0 {
        return Err(SemaError::new("lora `alpha` must be positive"));
    }
    if l.targets.is_empty() {
        return Err(SemaError::new(
            "lora `targets` must name at least one linear",
        ));
    }
    for t in &l.targets {
        if !LORA_TARGETS.contains(&t.as_str()) {
            return Err(SemaError::new(format!(
                "lora target `{t}` is not a base linear (expected one of {LORA_TARGETS:?})"
            )));
        }
    }
    Ok(l.clone())
}

fn req_int(model: &Model, key: &str) -> Result<i64, SemaError> {
    match model.config_value(key) {
        Some(Value::Int(n)) => Ok(*n),
        Some(_) => Err(SemaError::new(format!("config `{key}` must be an integer"))),
        None => Err(SemaError::new(format!(
            "missing required config field `{key}`"
        ))),
    }
}

fn opt_int(model: &Model, key: &str) -> Result<Option<i64>, SemaError> {
    match model.config_value(key) {
        Some(Value::Int(n)) => Ok(Some(*n)),
        Some(_) => Err(SemaError::new(format!("config `{key}` must be an integer"))),
        None => Ok(None),
    }
}

fn opt_bool(model: &Model, key: &str) -> Result<Option<bool>, SemaError> {
    match model.config_value(key) {
        Some(Value::Ident(s)) if s == "true" => Ok(Some(true)),
        Some(Value::Ident(s)) if s == "false" => Ok(Some(false)),
        Some(_) => Err(SemaError::new(format!(
            "config `{key}` must be `true` or `false`"
        ))),
        None => Ok(None),
    }
}

fn opt_float(model: &Model, key: &str) -> Result<Option<f64>, SemaError> {
    match model.config_value(key) {
        Some(Value::Float(x)) => Ok(Some(*x)),
        Some(Value::Int(n)) => Ok(Some(*n as f64)),
        Some(_) => Err(SemaError::new(format!("config `{key}` must be a number"))),
        None => Ok(None),
    }
}

fn resolve_config(model: &Model) -> Result<Config, SemaError> {
    let hidden = req_int(model, "hidden")?;
    let layers = req_int(model, "layers")?;
    let heads = req_int(model, "heads")?;
    let head_dim = req_int(model, "head_dim")?;
    let vocab = req_int(model, "vocab")?;

    let kv_heads = opt_int(model, "kv_heads")?.unwrap_or(heads);
    let ffn_dim = opt_int(model, "ffn_dim")?.unwrap_or(4 * hidden);
    let max_seq = opt_int(model, "max_seq")?.unwrap_or(2048);
    let rope_base = opt_float(model, "rope_base")?.unwrap_or(10000.0);
    let norm_eps = opt_float(model, "norm_eps")?.unwrap_or(1e-5);
    let attn_bias = opt_bool(model, "attn_bias")?.unwrap_or(false);

    for (name, v) in [
        ("hidden", hidden),
        ("layers", layers),
        ("heads", heads),
        ("head_dim", head_dim),
        ("vocab", vocab),
        ("kv_heads", kv_heads),
    ] {
        if v <= 0 {
            return Err(SemaError::new(format!("config `{name}` must be positive")));
        }
    }
    if heads % kv_heads != 0 {
        return Err(SemaError::new(format!(
            "kv_heads ({kv_heads}) must divide heads ({heads}) evenly"
        )));
    }

    let kv_dtype = match model.config_value("kv_dtype") {
        Some(Value::Ident(s)) => Some(parse_dtype(s)?),
        Some(_) => {
            return Err(SemaError::new(
                "config `kv_dtype` must be a dtype identifier",
            ))
        }
        None => None,
    };

    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let qkv_out = q_dim + 2 * kv_dim;
    let heads_per_group = heads / kv_heads;
    let attention = if kv_heads == heads {
        Attention::Mha
    } else if kv_heads == 1 {
        Attention::Mqa
    } else {
        Attention::Gqa
    };

    Ok(Config {
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        ffn_dim,
        vocab,
        max_seq,
        rope_base,
        norm_eps,
        kv_dtype,
        attn_bias,
        q_dim,
        kv_dim,
        qkv_out,
        heads_per_group,
        attention,
    })
}

fn resolve_weights(model: &Model) -> Result<Weights, SemaError> {
    let format = match model.weights_value("format") {
        Some(Value::Ident(s)) => match s.as_str() {
            "huggingface" => WeightFormat::HuggingFace,
            "gguf" => WeightFormat::Gguf,
            other => return Err(SemaError::new(format!("unknown weight format `{other}`"))),
        },
        Some(_) => return Err(SemaError::new("weights `format` must be an identifier")),
        None => return Err(SemaError::new("missing required weights field `format`")),
    };
    let source = match model.weights_value("source") {
        Some(Value::Str(s)) => s.clone(),
        Some(Value::Ident(s)) => s.clone(),
        _ => return Err(SemaError::new("missing required weights field `source`")),
    };
    let dtype = match model.weights_value("dtype") {
        Some(Value::Ident(s)) => parse_dtype(s)?,
        Some(_) => return Err(SemaError::new("weights `dtype` must be a dtype identifier")),
        None => return Err(SemaError::new("missing required weights field `dtype`")),
    };
    Ok(Weights {
        format,
        source,
        dtype,
    })
}

fn parse_dtype(s: &str) -> Result<Dtype, SemaError> {
    Ok(match s {
        "fp16" => Dtype::Fp16,
        "int8" => Dtype::Int8,
        "int4" => Dtype::Int4,
        "q4k" => Dtype::Q4k,
        "w3" => Dtype::W3,
        "gguf_q4km" => Dtype::GgufQ4km,
        other => return Err(SemaError::new(format!("unknown dtype `{other}`"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jac_syntax::parse_model;

    const TINY: &str = r#"
    model tinyllama {
      config { hidden = 2048  layers = 22  heads = 32  kv_heads = 4
               head_dim = 64  ffn_dim = 5632  vocab = 32000
               max_seq = 2048  rope_base = 10000.0  norm_eps = 1e-5 }
      weights { format = huggingface  source = "org/tinyllama"  dtype = fp16 }
      layer decoder { x = rmsnorm(input, w, norm_eps) }
      pipeline { embed(t, e) -> greedy_sample }
    }"#;

    #[test]
    fn derives_tinyllama_dims() {
        let ir = lower(&parse_model(TINY).unwrap()).unwrap();
        let c = &ir.config;
        assert_eq!(c.kv_dim, 256); // 4 * 64
        assert_eq!(c.q_dim, 2048); // 32 * 64
        assert_eq!(c.qkv_out, 2560); // 2048 + 256 + 256
        assert_eq!(c.heads_per_group, 8); // 32 / 4
        assert_eq!(c.attention, Attention::Gqa);
        assert_eq!(ir.weights.dtype, Dtype::Fp16);
    }

    #[test]
    fn applies_defaults() {
        let src = r#"model m {
          config { hidden = 512  layers = 4  heads = 8  head_dim = 64  vocab = 1000 }
          weights { format = huggingface  source = "x"  dtype = fp16 }
          layer d { x = rmsnorm(input, w, eps) }
          pipeline { greedy_sample }
        }"#;
        let ir = lower(&parse_model(src).unwrap()).unwrap();
        assert_eq!(ir.config.kv_heads, 8); // defaults to heads -> MHA
        assert_eq!(ir.config.attention, Attention::Mha);
        assert_eq!(ir.config.ffn_dim, 2048); // 4 * hidden
        assert_eq!(ir.config.max_seq, 2048);
    }

    #[test]
    fn derives_qwen2_dims_with_attn_bias() {
        let src = r#"model qwen2 {
          config { hidden = 896  layers = 24  heads = 14  kv_heads = 2  head_dim = 64
                   ffn_dim = 4864  vocab = 151936  max_seq = 2048
                   rope_base = 1000000.0  norm_eps = 1e-6  attn_bias = true }
          weights { format = huggingface  source = "Qwen/Qwen2-0.5B-Instruct"  dtype = fp16 }
          layer decoder { x = rmsnorm(input, w, norm_eps) }
          pipeline { embed(t, e) -> greedy_sample }
        }"#;
        let ir = lower(&parse_model(src).unwrap()).unwrap();
        let c = &ir.config;
        assert_eq!(c.q_dim, 896); // 14 * 64
        assert_eq!(c.kv_dim, 128); // 2 * 64
        assert_eq!(c.qkv_out, 1152); // 896 + 128 + 128
        assert_eq!(c.heads_per_group, 7); // 14 / 2
        assert_eq!(c.attention, Attention::Gqa);
        assert!(c.attn_bias, "qwen2 should parse attn_bias = true");
    }

    #[test]
    fn attn_bias_defaults_false() {
        let ir = lower(&parse_model(TINY).unwrap()).unwrap();
        assert!(
            !ir.config.attn_bias,
            "attn_bias should default to false for LLaMA-family"
        );
    }

    #[test]
    fn validates_lora_block() {
        let src = r#"model m {
          config { hidden = 8 layers = 1 heads = 2 head_dim = 4 vocab = 10 }
          weights { format = huggingface source = "x" dtype = fp16 }
          layer d { x = rmsnorm(a, b, c) }
          pipeline { greedy_sample }
          lora { rank = 16 alpha = 32 targets = w_q, w_v source = "org/a" }
        }"#;
        let ir = lower(&parse_model(src).unwrap()).unwrap();
        let l = ir.lora.expect("lora should be carried on the IR");
        assert_eq!(l.rank, 16);
        assert_eq!(l.targets, vec!["w_q", "w_v"]);
    }

    #[test]
    fn rejects_unknown_lora_target() {
        let src = r#"model m {
          config { hidden = 8 layers = 1 heads = 2 head_dim = 4 vocab = 10 }
          weights { format = huggingface source = "x" dtype = fp16 }
          layer d { x = rmsnorm(a, b, c) }
          pipeline { greedy_sample }
          lora { rank = 8 alpha = 16 targets = w_q, w_mlp source = "org/a" }
        }"#;
        let err = lower(&parse_model(src).unwrap()).unwrap_err();
        assert!(err.message.contains("w_mlp"), "got: {}", err.message);
    }

    #[test]
    fn lowers_experiment_and_expands_sweep() {
        let src = r#"experiment e {
          kind = kernel  fixture { op = "specs/gemv.jac" }
          subjects { naive = jac(naive)  cublas = jac(cublas) }
          reference = cublas
          sweep { M = [1024, 4096]  N = [128, 512] }
          metrics = [correctness, latency, memory] }"#;
        let e = jac_syntax::parse_experiment(src).unwrap();
        let ir = lower_experiment(&e).unwrap();
        assert_eq!(ir.op_path, "specs/gemv.jac");
        // cartesian product M×N = 2×2 = 4 scenarios, deterministic seeds, stable ids.
        assert_eq!(ir.scenarios.len(), 4);
        assert_eq!(
            ir.scenarios[0].defines,
            vec![("M".into(), 1024), ("N".into(), 128)]
        );
        assert_eq!(ir.scenarios[0].id, "M1024_N128");
        assert!(
            ir.scenarios
                .iter()
                .map(|s| s.seed)
                .collect::<std::collections::HashSet<_>>()
                .len()
                == 4
        );
    }

    #[test]
    fn rejects_reference_not_a_subject() {
        let src = r#"experiment e { kind = kernel  fixture { op = "k.jac" }
          subjects { a = jac(naive) }  reference = ghost  sweep { M = [1] }  metrics = [latency] }"#;
        let e = jac_syntax::parse_experiment(src).unwrap();
        assert!(lower_experiment(&e)
            .unwrap_err()
            .message
            .contains("reference"));
    }

    #[test]
    fn rejects_kernel_experiment_without_op() {
        let src = r#"experiment e { kind = kernel  fixture { }
          subjects { a = jac(naive) }  reference = a  sweep { M = [1] }  metrics = [latency] }"#;
        let e = jac_syntax::parse_experiment(src).unwrap();
        assert!(lower_experiment(&e).unwrap_err().message.contains("op"));
    }

    #[test]
    fn rejects_indivisible_kv_heads() {
        let src = r#"model m {
          config { hidden = 512 layers = 1 heads = 8 kv_heads = 3 head_dim = 64 vocab = 10 }
          weights { format = huggingface source = "x" dtype = fp16 }
          layer d { x = rmsnorm(a, b, c) }
          pipeline { greedy_sample }
        }"#;
        let err = lower(&parse_model(src).unwrap()).unwrap_err();
        assert!(err.message.contains("must divide heads"));
    }
}
