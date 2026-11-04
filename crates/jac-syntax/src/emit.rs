//! Serialize a [`Model`] AST back to `.jac` source text.
//!
//! Used by model surgery to write a transformed spec that re-parses and re-lowers identically to
//! the in-memory model. Comments and original whitespace are not preserved — this is generator
//! output, not a formatter for hand-written files.

use crate::ast::{BinOp, Expr, Model, Value};

/// Render `model` as a `.jac` source string.
pub fn model_to_jac(model: &Model) -> String {
    let mut s = String::new();
    s.push_str(&format!("model {} {{\n", model.name));

    s.push_str("\n  config {\n");
    for (k, v) in &model.config {
        s.push_str(&format!("    {k} = {}\n", value_to_jac(v)));
    }
    s.push_str("  }\n");

    s.push_str("\n  weights {\n");
    for (k, v) in &model.weights {
        s.push_str(&format!("    {k} = {}\n", value_to_jac(v)));
    }
    s.push_str("  }\n");

    for layer in &model.layers {
        s.push_str(&format!("\n  layer {} {{\n", layer.name));
        for stmt in &layer.stmts {
            s.push_str(&format!(
                "    {} = {}\n",
                stmt.targets.join(", "),
                expr_to_jac(&stmt.value)
            ));
        }
        s.push_str("  }\n");
    }

    if !model.pipeline.is_empty() {
        s.push_str("\n  pipeline {\n    ");
        let steps: Vec<String> = model.pipeline.iter().map(expr_to_jac).collect();
        s.push_str(&steps.join("\n    -> "));
        s.push_str("\n  }\n");
    }

    if let Some(l) = &model.lora {
        s.push_str(&format!(
            "\n  lora {{\n    rank    = {}\n    alpha   = {}\n    targets = {}\n    source  = {:?}\n  }}\n",
            l.rank,
            float_to_jac(l.alpha),
            l.targets.join(", "),
            l.source,
        ));
    }

    s.push_str("}\n");
    s
}

fn value_to_jac(v: &Value) -> String {
    match v {
        Value::Int(n) => n.to_string(),
        Value::Float(x) => float_to_jac(*x),
        Value::Ident(s) => s.clone(),
        Value::Str(s) => format!("{s:?}"),
    }
}

fn expr_to_jac(e: &Expr) -> String {
    match e {
        Expr::Ident(s) => s.clone(),
        Expr::Int(n) => n.to_string(),
        Expr::Float(x) => float_to_jac(*x),
        Expr::Str(s) => format!("{s:?}"),
        Expr::Call { name, args } => {
            let args: Vec<String> = args.iter().map(expr_to_jac).collect();
            format!("{name}({})", args.join(", "))
        }
        Expr::Binary { op, lhs, rhs } => {
            let op = match op {
                BinOp::Add => "+",
                BinOp::Mul => "*",
            };
            format!("{} {op} {}", expr_to_jac(lhs), expr_to_jac(rhs))
        }
    }
}

/// Format an f64 so it re-lexes as a float (Rust's `{:?}` keeps a decimal point or exponent,
/// e.g. `10000.0` and `1e-5`, rather than the integer-looking `{}` output).
fn float_to_jac(x: f64) -> String {
    format!("{x:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_model;

    const SRC: &str = r#"
    model tiny {
      config { hidden = 2048  layers = 22  rope_base = 10000.0  norm_eps = 1e-5 }
      weights { format = huggingface  source = "org/model"  dtype = fp16 }
      layer decoder {
        normed = rmsnorm(input, norm_weight, norm_eps)
        q, k, v = linear(normed, w_q, w_k, w_v)
        ffn_mid = silu(gate) * up
      }
      pipeline { embed(token, e) -> repeat(decoder, layers) -> greedy_sample }
    }"#;

    #[test]
    fn round_trips_through_parse() {
        let original = parse_model(SRC).unwrap();
        let text = model_to_jac(&original);
        let reparsed =
            parse_model(&text).unwrap_or_else(|e| panic!("re-parse failed: {}", e.message));
        assert_eq!(
            original, reparsed,
            "model_to_jac output must re-parse to the same AST"
        );
    }

    #[test]
    fn preserves_float_and_fused_targets() {
        let text = model_to_jac(&parse_model(SRC).unwrap());
        assert!(text.contains("rope_base = 10000.0"));
        assert!(text.contains("q, k, v = linear(normed, w_q, w_k, w_v)"));
        assert!(text.contains("silu(gate) * up"));
        assert!(text.contains("-> repeat(decoder, layers)"));
    }
}
