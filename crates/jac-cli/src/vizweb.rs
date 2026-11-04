//! `--viz-web`: render a self-contained interactive HTML explorer for a model spec.
//!
//! Produces `<name>_viz.html` with no external dependencies (inline CSS/JS): the config and
//! derived dimensions, the pipeline as a flow, the decoder block's operations, and an interactive
//! memory estimate whose weight footprint and KV-cache size recompute live as the viewer changes
//! the dtype and sequence length. Built from the lowered IR, so it always matches what the
//! compiler would generate.

use std::path::PathBuf;

use jac_ir::{Attention, ModelIr, WeightFormat};
use jac_syntax::ast::{BinOp, Expr};

/// Run `jacq --viz-web <spec>`: write `<name>_viz.html` into the current directory.
pub fn run(ir: &ModelIr) -> std::process::ExitCode {
    let html = render(ir);
    let out = PathBuf::from(format!("{}_viz.html", ir.name));
    match std::fs::write(&out, html) {
        Ok(()) => {
            println!("\u{2713} wrote {}  (open in a browser)", out.display());
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("jacq --viz-web: cannot write {}: {e}", out.display());
            std::process::ExitCode::FAILURE
        }
    }
}

fn render(ir: &ModelIr) -> String {
    let c = &ir.config;
    let attn = match c.attention {
        Attention::Mha => "MHA",
        Attention::Gqa => "GQA",
        Attention::Mqa => "MQA",
    };
    let fmt = match ir.weights.format {
        WeightFormat::HuggingFace => "huggingface",
        WeightFormat::Gguf => "gguf",
    };
    let dtype = format!("{:?}", ir.weights.dtype).to_lowercase();

    // Parameter accounting (counts, dtype-independent). The four large linears are what scale with
    // the chosen dtype; embedding, lm_head, and norms stay fp16 in the generated runtime.
    let per_layer_linear =
        c.qkv_out * c.hidden + c.hidden * c.q_dim + 2 * c.ffn_dim * c.hidden + c.hidden * c.ffn_dim;
    let linear_params = c.layers * per_layer_linear;
    let fp16_params = c.vocab * c.hidden          // embedding
        + c.vocab * c.hidden                      // lm_head (≈ embedding when tied)
        + c.layers * 2 * c.hidden                 // attn + ffn norms
        + c.hidden                                // final norm
        + if c.attn_bias { c.layers * c.qkv_out } else { 0 };
    // KV cache bytes per cached token (fp16): both K and V, across layers and kv-heads.
    let kv_bytes_per_token = 2 * c.layers * c.kv_heads * c.head_dim * 2;

    let config_rows = [
        ("hidden", c.hidden.to_string()),
        ("layers", c.layers.to_string()),
        ("heads", c.heads.to_string()),
        ("kv_heads", c.kv_heads.to_string()),
        ("head_dim", c.head_dim.to_string()),
        ("ffn_dim", c.ffn_dim.to_string()),
        ("vocab", c.vocab.to_string()),
        ("max_seq", c.max_seq.to_string()),
        ("rope_base", c.rope_base.to_string()),
        ("norm_eps", c.norm_eps.to_string()),
        ("attn_bias", c.attn_bias.to_string()),
    ]
    .iter()
    .map(|(k, v)| format!("<tr><td>{k}</td><td>{}</td></tr>", esc(v)))
    .collect::<String>();

    let derived_rows = [
        ("q_dim", c.q_dim.to_string()),
        ("kv_dim", c.kv_dim.to_string()),
        ("qkv_out", c.qkv_out.to_string()),
        ("heads_per_group", c.heads_per_group.to_string()),
        ("attention", attn.to_string()),
    ]
    .iter()
    .map(|(k, v)| format!("<tr><td>{k}</td><td>{}</td></tr>", esc(v)))
    .collect::<String>();

    let pipeline = pipeline_html(ir);
    let layer_ops = layer_ops_html(ir);

    TEMPLATE
        .replace("__NAME__", &esc(&ir.name))
        .replace("__ARCH__", attn)
        .replace("__FMT__", fmt)
        .replace("__DTYPE__", &dtype)
        .replace("__SOURCE__", &esc(&ir.weights.source))
        .replace("__CONFIG_ROWS__", &config_rows)
        .replace("__DERIVED_ROWS__", &derived_rows)
        .replace("__PIPELINE__", &pipeline)
        .replace("__LAYER_OPS__", &layer_ops)
        .replace("__LINEAR_PARAMS__", &linear_params.to_string())
        .replace("__FP16_PARAMS__", &fp16_params.to_string())
        .replace("__KV_PER_TOKEN__", &kv_bytes_per_token.to_string())
        .replace("__MAX_SEQ__", &c.max_seq.to_string())
        .replace("__DEFAULT_DTYPE__", &dtype)
}

/// The pipeline as a row of flow boxes (`embed → decoder ×N → … → greedy_sample`).
fn pipeline_html(ir: &ModelIr) -> String {
    let steps: Vec<String> = ir
        .model
        .pipeline
        .iter()
        .map(|step| match step {
            Expr::Ident(s) => s.clone(),
            Expr::Call { name, args } if name == "repeat" => {
                let layer = match args.first() {
                    Some(Expr::Ident(s)) => s.as_str(),
                    _ => "layer",
                };
                let count = match args.get(1) {
                    Some(Expr::Int(n)) => *n,
                    Some(Expr::Ident(s)) if s == "layers" => ir.config.layers,
                    _ => 0,
                };
                format!("{layer} \u{00d7}{count}")
            }
            Expr::Call { name, .. } => name.clone(),
            _ => "?".to_string(),
        })
        .collect();
    steps
        .iter()
        .map(|s| format!("<span class=\"node\">{}</span>", esc(s)))
        .collect::<Vec<_>>()
        .join("<span class=\"arrow\">\u{2192}</span>")
}

/// The first layer block's statements as a vertical op list (`targets = expr`).
fn layer_ops_html(ir: &ModelIr) -> String {
    let Some(layer) = ir.model.layers.first() else {
        return "<li><em>no layer block</em></li>".into();
    };
    layer
        .stmts
        .iter()
        .map(|stmt| {
            format!(
                "<li><code class=\"lhs\">{}</code> = <code>{}</code></li>",
                esc(&stmt.targets.join(", ")),
                esc(&expr_str(&stmt.value))
            )
        })
        .collect()
}

/// Compact source-like rendering of a layer expression (mirrors the `.jac` form).
fn expr_str(e: &Expr) -> String {
    match e {
        Expr::Ident(s) => s.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(x) => format!("{x:?}"),
        Expr::Str(s) => format!("{s:?}"),
        Expr::Call { name, args } => {
            format!(
                "{name}({})",
                args.iter().map(expr_str).collect::<Vec<_>>().join(", ")
            )
        }
        Expr::Binary { op, lhs, rhs } => {
            let op = match op {
                BinOp::Add => "+",
                BinOp::Mul => "*",
            };
            format!("{} {op} {}", expr_str(lhs), expr_str(rhs))
        }
    }
}

/// Minimal HTML-escape for text injected into the page.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const TEMPLATE: &str = include_str!("templates/viz.html");

#[cfg(test)]
mod tests {
    use super::*;

    fn ir() -> ModelIr {
        let src = r#"model tiny {
          config { hidden = 2048 layers = 22 heads = 32 kv_heads = 4 head_dim = 64
                   ffn_dim = 5632 vocab = 32000 max_seq = 2048 rope_base = 10000.0 norm_eps = 1e-5 }
          weights { format = huggingface source = "org/tiny" dtype = int8 }
          layer decoder { normed = rmsnorm(input, w, norm_eps)  ffn = silu(gate) * up }
          pipeline { embed(t, e) -> repeat(decoder, layers) -> lm_head(l) -> greedy_sample }
        }"#;
        jac_ir::lower(&jac_syntax::parse_model(src).unwrap()).unwrap()
    }

    #[test]
    fn renders_without_leftover_placeholders() {
        let html = render(&ir());
        // A missed `.replace` would leave a `__NAME__`-style token in the page.
        assert!(
            !html.contains("__"),
            "viz HTML has an unsubstituted placeholder"
        );
    }

    #[test]
    fn renders_key_content() {
        let html = render(&ir());
        assert!(
            html.contains("decoder \u{00d7}22"),
            "pipeline repeat count missing"
        );
        assert!(html.contains("GQA"), "architecture badge missing");
        assert!(
            html.contains("silu(gate) * up"),
            "layer op rendering missing"
        );
        assert!(html.contains("<title>jacq · tiny</title>"));
        // int8 spec → the dtype default injected into the page's JS is "int8".
        assert!(
            html.contains("const want = \"int8\""),
            "dtype default not substituted"
        );
    }

    #[test]
    fn escapes_angle_brackets() {
        // Guard against HTML injection via spec strings.
        assert_eq!(esc("a<b>&c"), "a&lt;b&gt;&amp;c");
    }
}
