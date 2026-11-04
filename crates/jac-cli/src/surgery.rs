//! Model surgery: declarative `.jac` → `.jac` transformations.
//!
//! Phase 2 implements `prune layers`, `cast weights to <dtype>`, `update config { ... }`, and
//! `merge layers ... from "donor.jac"`, which compose in a single surgery block. Each surgery
//! emits a transformed model spec plus a Python weight-conversion script that renumbers kept
//! layers, optionally re-quantizes (fp16 → int8/int4) with the exporter's exact formula, and pulls
//! merged layers from a donor weight dir. The output spec flows through the existing
//! `--emit-model` / `--run` pipeline unchanged.

use std::path::{Path, PathBuf};

use jac_syntax::ast::{SurgeryOp, Value};
use jac_syntax::{model_to_jac, parse_model, parse_surgery};

/// Run `jacq --surgery <file>`: read a surgery spec, apply it, and write `<name>.jac` +
/// `<name>_convert.py` into the current directory.
pub fn run(path: &Path) -> std::process::ExitCode {
    match run_inner(path) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("jacq --surgery: {msg}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run_inner(path: &Path) -> Result<(), String> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let surgery = parse_surgery(&src).map_err(|e| format!("{}: {}", path.display(), e.message))?;

    // Resolve the source model path (relative to the surgery file, or project root for `@`).
    let from_path = resolve_from(&surgery.from, path)?;
    let (mut model, ir) = load_model(&from_path)?;
    let orig_layers = ir.config.layers;
    let src_dtype = format!("{:?}", ir.weights.dtype).to_lowercase();

    // Apply ops in order. They compose: prune narrows layers, update overrides config, cast
    // changes the weight dtype. `kept` and `target_dtype` accumulate across ops.
    let mut kept: Vec<i64> = (0..orig_layers).collect();
    let mut target_dtype: Option<String> = None;
    let mut merge_set: Vec<i64> = Vec::new();
    let mut merge_donor: Option<String> = None;
    let mut summary: Vec<String> = Vec::new();
    for op in &surgery.ops {
        match op {
            SurgeryOp::PruneLayers(ranges) => {
                let remove = collect_removed(ranges, orig_layers)?;
                kept.retain(|i| !remove.contains(i));
                summary.push(format!("prune {} layer(s)", remove.len()));
            }
            SurgeryOp::CastWeights(dt) => {
                let dt = dt.to_lowercase();
                if !matches!(dt.as_str(), "fp16" | "int8" | "int4") {
                    return Err(format!(
                        "cast target dtype `{dt}` unsupported (expected fp16, int8, or int4)"
                    ));
                }
                target_dtype = Some(dt);
            }
            SurgeryOp::UpdateConfig(kvs) => {
                for (k, v) in kvs {
                    set_kv(&mut model.config, k, v.clone());
                }
                summary.push(format!("update {} config field(s)", kvs.len()));
            }
            SurgeryOp::ApplyLora(spec) => {
                model.lora = Some(spec.clone());
                summary.push(format!(
                    "apply lora (rank {}, {} targets)",
                    spec.rank,
                    spec.targets.len()
                ));
            }
            SurgeryOp::MergeLayers { ranges, donor } => {
                match &merge_donor {
                    Some(d) if d != donor => {
                        return Err(
                            "multiple distinct donor models in one surgery are not yet supported"
                                .into(),
                        );
                    }
                    _ => merge_donor = Some(donor.clone()),
                }
                for i in collect_removed(ranges, orig_layers)? {
                    if !merge_set.contains(&i) {
                        merge_set.push(i);
                    }
                }
                summary.push(format!("merge {} layer(s) from donor", merge_set.len()));
            }
        }
    }
    if kept.is_empty() {
        return Err("surgery would remove every layer".into());
    }
    let new_layers = kept.len() as i64;

    // Validate the donor (if any) is shape- and dtype-compatible: its weight files are copied
    // directly into the merged layer slots, so the linear shapes and weight format must match.
    if let Some(donor_rel) = &merge_donor {
        let donor_path = resolve_from(donor_rel, path)?;
        let (_, donor_ir) = load_model(&donor_path)?;
        check_donor_compat(&ir, &donor_ir, &merge_set)?;
    }

    // Rewrite the model: new name, reduced layer count, possibly new dtype.
    model.name = surgery.name.clone();
    set_kv(&mut model.config, "layers", Value::Int(new_layers));
    let dst_dtype = target_dtype.clone().unwrap_or_else(|| src_dtype.clone());
    if let Some(t) = &target_dtype {
        if *t != src_dtype && src_dtype != "fp16" {
            return Err(format!(
                "cast currently supports casting from fp16 (source dtype is {src_dtype}); \
                 dequantize-then-requantize is not yet implemented"
            ));
        }
        set_kv(&mut model.weights, "dtype", Value::Ident(t.clone()));
        summary.push(format!("cast {src_dtype} \u{2192} {t}"));
    }
    // Re-quantization is only needed for fp16 -> int8/int4; otherwise the convert script copies.
    let requant = dst_dtype != src_dtype
        && src_dtype == "fp16"
        && matches!(dst_dtype.as_str(), "int8" | "int4");
    let ops_desc = if summary.is_empty() {
        "no-op".to_string()
    } else {
        summary.join(", ")
    };

    // Write the new spec + the weight-conversion script into the current directory.
    let jac_path = PathBuf::from(format!("{}.jac", surgery.name));
    let py_path = PathBuf::from(format!("{}_convert.py", surgery.name));
    let mut spec = format!(
        "// Generated by jacq surgery `{}` ({ops_desc}). Regenerate; do not hand-edit.\n",
        surgery.name
    );
    spec.push_str(&model_to_jac(&model));
    std::fs::write(&jac_path, &spec)
        .map_err(|e| format!("cannot write {}: {e}", jac_path.display()))?;
    merge_set.sort_unstable();
    std::fs::write(
        &py_path,
        convert_script(
            &surgery.name,
            &kept,
            &dst_dtype,
            requant,
            &merge_set,
            &ir.config,
        ),
    )
    .map_err(|e| format!("cannot write {}: {e}", py_path.display()))?;

    println!("\u{2713} surgery `{}`: {ops_desc}", surgery.name);
    if new_layers != orig_layers {
        println!("  layers {orig_layers} \u{2192} {new_layers} (kept {kept:?})");
    }
    println!("\u{2713} wrote {}", jac_path.display());
    let donor_arg = if merge_set.is_empty() {
        ""
    } else {
        " <donor_weights>"
    };
    println!(
        "\u{2713} wrote {}  (run: python3 {} <src_weights> <dst_weights>{donor_arg})",
        py_path.display(),
        py_path.display()
    );
    Ok(())
}

/// Validate a donor model is compatible for `merge`: identical linear shapes (so weight files
/// drop in), matching weight dtype and attention-bias, and enough layers to cover the merge set.
fn check_donor_compat(
    src: &jac_ir::ModelIr,
    donor: &jac_ir::ModelIr,
    merge_set: &[i64],
) -> Result<(), String> {
    let (a, b) = (&src.config, &donor.config);
    let dims = [
        ("hidden", a.hidden, b.hidden),
        ("head_dim", a.head_dim, b.head_dim),
        ("q_dim", a.q_dim, b.q_dim),
        ("kv_dim", a.kv_dim, b.kv_dim),
        ("qkv_out", a.qkv_out, b.qkv_out),
        ("ffn_dim", a.ffn_dim, b.ffn_dim),
    ];
    for (name, sv, dv) in dims {
        if sv != dv {
            return Err(format!(
                "donor `{name}` ({dv}) does not match source ({sv}); cannot merge layers"
            ));
        }
    }
    if src.weights.dtype != donor.weights.dtype {
        return Err(format!(
            "donor dtype ({:?}) does not match source ({:?}); merge copies weight files directly",
            donor.weights.dtype, src.weights.dtype
        ));
    }
    if a.attn_bias != b.attn_bias {
        return Err(
            "donor attn_bias does not match source; merged layers would be missing/extra bias"
                .into(),
        );
    }
    if let Some(&max) = merge_set.iter().max() {
        if max >= b.layers {
            return Err(format!(
                "donor has {} layers but merge references index {max}",
                b.layers
            ));
        }
    }
    Ok(())
}

/// Read, parse, and lower a model spec. Returns `(ast_model, lowered_ir)` or a human-readable
/// error. Used for both the surgery source model and the donor model to avoid repeating the
/// read → parse_model → jac_ir::lower sequence at each call site.
fn load_model(path: &Path) -> Result<(jac_syntax::ast::Model, jac_ir::ModelIr), String> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let model = parse_model(&src).map_err(|e| format!("{}: {}", path.display(), e.message))?;
    let ir = jac_ir::lower(&model).map_err(|e| format!("{}: {}", path.display(), e.message))?;
    Ok((model, ir))
}

/// Expand inclusive ranges into the set of layer indices to remove, validating bounds.
fn collect_removed(ranges: &[jac_syntax::LayerRange], layers: i64) -> Result<Vec<i64>, String> {
    let mut remove = Vec::new();
    for r in ranges {
        if r.start < 0 || r.end >= layers {
            return Err(format!(
                "prune range {}..{} is out of bounds for a {}-layer model (valid 0..{})",
                r.start,
                r.end,
                layers,
                layers - 1
            ));
        }
        for i in r.start..=r.end {
            if !remove.contains(&i) {
                remove.push(i);
            }
        }
    }
    Ok(remove)
}

/// Resolve a surgery `from` path. `@`-prefixed paths are relative to the project root (nearest
/// ancestor containing `.git`); others are relative to the surgery spec file's directory.
fn resolve_from(from: &str, surgery_file: &Path) -> Result<PathBuf, String> {
    if let Some(rest) = from.strip_prefix('@') {
        let start = surgery_file.parent().unwrap_or_else(|| Path::new("."));
        let root = project_root(start).ok_or_else(|| {
            format!(
                "`@{rest}`: no project root (.git) found above {}",
                start.display()
            )
        })?;
        Ok(root.join(rest))
    } else {
        let base = surgery_file.parent().unwrap_or_else(|| Path::new("."));
        Ok(base.join(from))
    }
}

/// Walk up from `start` to the nearest directory containing a `.git` entry.
fn project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.canonicalize().ok()?;
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Set (or append) a `key = value` entry, preserving source order for existing keys. Works on
/// both the config and weights blocks (each a `Vec<(String, Value)>`).
fn set_kv(entries: &mut Vec<(String, Value)>, key: &str, value: Value) {
    if let Some(entry) = entries.iter_mut().find(|(k, _)| k == key) {
        entry.1 = value;
    } else {
        entries.push((key.to_string(), value));
    }
}

/// Build the weight-conversion script. It always renumbers the kept layers into a contiguous
/// range (copying files, dtype-agnostic via globbing). When `requant` is set (fp16 → int8/int4),
/// the four linear projections are re-quantized in place using the same per-output-row symmetric
/// formula as the generated exporter, so the result is byte-identical to a fresh quantized export.
fn convert_script(
    name: &str,
    kept: &[i64],
    dst_dtype: &str,
    requant: bool,
    merge_set: &[i64],
    c: &jac_ir::Config,
) -> String {
    let kept_list = kept
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let merge_list = merge_set
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let requant = if requant { "True" } else { "False" }; // Python bool literal
                                                          // Per-linear matrix shapes [out, in], needed to reshape a flat fp16 .bin for per-row quant.
    let shapes = format!(
        "{{\"qkv\": ({}, {}), \"o\": ({}, {}), \"gate_up\": ({}, {}), \"down\": ({}, {})}}",
        c.qkv_out,
        c.hidden,
        c.hidden,
        c.q_dim,
        2 * c.ffn_dim,
        c.hidden,
        c.hidden,
        c.ffn_dim
    );
    format!(
        r#"#!/usr/bin/env python3
# Weight conversion for surgery `{name}` (target dtype {dst_dtype}). Renumbers the kept layers
# into a contiguous range; when re-quantizing (fp16 -> int8/int4) the four linear projections
# (qkv, o, gate_up, down) are quantized per-output-row, matching the generated exporter exactly.
import glob
import os
import shutil
import sys

import numpy as np

KEPT = [{kept_list}]                      # original layer indices to keep, in order
MERGE = [{merge_list}]                     # original indices whose weights come from the donor dir
DST_DTYPE = "{dst_dtype}"
REQUANT = {requant}                       # fp16 -> int8/int4 re-quantization needed
SHAPES = {shapes}                         # [out, in] per linear, for reshape before per-row quant
LINEARS = ("qkv", "o", "gate_up", "down")  # quantizable; norms / qkv_bias stay fp16


def quant_int8(W):
    scale = np.clip(np.abs(W).max(axis=1, keepdims=True) / 127.0, 1e-8, None)  # [out, 1]
    Wq = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return Wq, scale.squeeze(1).astype(np.float16)


def quant_int4(W):
    assert W.shape[1] % 2 == 0, "in_dim must be even to pack int4"
    scale = np.clip(np.abs(W).max(axis=1, keepdims=True) / 7.0, 1e-8, None)
    q = np.clip(np.round(W / scale), -7, 7).astype(np.int8)
    lo = (q[:, 0::2] & 0x0F).astype(np.uint8)
    hi = (q[:, 1::2] & 0x0F).astype(np.uint8)
    return (lo | (hi << 4)).astype(np.uint8), scale.squeeze(1).astype(np.float16)


def requant_linear(src_path, dst_base, kind):
    out_dim, in_dim = SHAPES[kind]
    W = np.fromfile(src_path, dtype=np.float16).astype(np.float32).reshape(out_dim, in_dim)
    packed, scale = (quant_int4(W) if DST_DTYPE == "int4" else quant_int8(W))
    packed.tofile(dst_base + ".bin")
    scale.tofile(dst_base + ".scale.bin")


def linear_kind(filename):
    # filename is `layerNN_<kind>.bin`; return <kind> if it's a quantizable linear, else None.
    for kind in LINEARS:
        if filename.endswith(f"_{{kind}}.bin"):
            return kind
    return None


def main():
    need_donor = len(MERGE) > 0
    if len(sys.argv) < (4 if need_donor else 3):
        donor = " <donor_weight_dir>" if need_donor else ""
        print(f"usage: {{sys.argv[0]}} <src_weight_dir> <dst_weight_dir>{{donor}}")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    donor = sys.argv[3] if need_donor else None
    os.makedirs(dst, exist_ok=True)
    # Non-layer weights always come from the source model.
    for name in ("embed.bin", "final_norm.bin", "lm_head.bin", "test_tokens.bin"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, name))
    for new_i, old_i in enumerate(KEPT):
        layer_src = donor if old_i in MERGE else src   # merged layers pulled from the donor model
        for f in glob.glob(os.path.join(layer_src, f"layer{{old_i:02d}}_*")):
            base = os.path.basename(f).replace(f"layer{{old_i:02d}}_", f"layer{{new_i:02d}}_", 1)
            kind = linear_kind(os.path.basename(f)) if REQUANT else None
            if kind is not None:
                requant_linear(f, os.path.join(dst, base[:-4]), kind)  # strip ".bin"
            else:
                shutil.copy2(f, os.path.join(dst, base))
    print(f"converted {{len(KEPT)}} layers -> {{dst}} (dtype={{DST_DTYPE}}, requant={{REQUANT}}, merged={{len(MERGE)}})")


if __name__ == "__main__":
    main()
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(hidden: i64, layers: i64) -> jac_ir::ModelIr {
        let src = format!(
            "model m {{ config {{ hidden = {hidden} layers = {layers} heads = 8 kv_heads = 8 \
             head_dim = {hd} ffn_dim = {ffn} vocab = 100 }} \
             weights {{ format = huggingface source = \"x\" dtype = fp16 }} \
             layer d {{ x = rmsnorm(input, w, eps) }} pipeline {{ greedy_sample }} }}",
            hd = hidden / 8,
            ffn = 4 * hidden,
        );
        jac_ir::lower(&parse_model(&src).unwrap()).unwrap()
    }

    #[test]
    fn donor_compat_accepts_identical_dims() {
        let m = ir(512, 12);
        assert!(check_donor_compat(&m, &ir(512, 12), &[3, 4]).is_ok());
    }

    #[test]
    fn donor_compat_rejects_dim_mismatch() {
        let err = check_donor_compat(&ir(512, 12), &ir(256, 12), &[3]).unwrap_err();
        assert!(err.contains("hidden"), "got: {err}");
    }

    #[test]
    fn donor_compat_rejects_insufficient_layers() {
        let err = check_donor_compat(&ir(512, 12), &ir(512, 4), &[7]).unwrap_err();
        assert!(err.contains("index 7"), "got: {err}");
    }
}
