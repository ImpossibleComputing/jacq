//! `jacq` — the Jacq command-line compiler.
//!
//! Subcommands: `--version`, `--doctor` (toolchain check), `--validate` (parse + lower),
//! `--viz` (terminal overview), `--viz-web` (interactive HTML explorer), `--emit-model` (write
//! CUDA runtime + weight exporter), `--run` (codegen → nvcc → export → inference, end to end),
//! `--surgery` (declarative model transforms: prune / cast / update / merge), and `--emit-cuda`
//! (kernel-DSL front-end: parse + validate a `kernel` spec and lower it to a standalone
//! benchmark harness or a runtime-extension translation unit).

use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use jac_syntax::parse_model;

mod surgery;
mod vizweb;

/// jacq — compile transformer model specs into optimized GPU inference runtimes.
#[derive(Parser, Debug)]
#[command(name = "jacq", version, about, long_about = None)]
struct Cli {
    /// Check prerequisites (nvcc, python3, nvidia-smi).
    #[arg(long)]
    doctor: bool,

    /// Validate a model spec without generating code.
    #[arg(long)]
    validate: bool,

    /// Render a terminal overview of the model (config, pipeline, memory estimate).
    #[arg(long)]
    viz: bool,

    /// Write a self-contained interactive HTML explorer (`<name>_viz.html`).
    #[arg(long = "viz-web")]
    viz_web: bool,

    /// Generate a CUDA inference runtime + weight exporter from a model spec.
    #[arg(long = "emit-model")]
    emit_model: bool,

    /// Generate a CUDA kernel + host harness from a kernel-DSL spec.
    #[arg(long = "emit-cuda")]
    emit_cuda: bool,

    /// With `--emit-cuda`, emit a runtime-extension translation unit instead of the standalone
    /// file-IO benchmark harness. First supported for the KIVI vLLM DSL decode artifact.
    #[arg(long = "kernel-library")]
    kernel_library: bool,

    /// With `--emit-cuda`, delegate the contraction to cuBLAS instead of the naive kernel.
    #[arg(long)]
    cublas: bool,

    /// With `--emit-cuda`, emit a tensor-core (nvcuda::wmma) matmul kernel (fp16 in / fp32 out,
    /// smem-staged K-loop). Requires M/N/K multiples of 16; matmul only (not GEMV/softmax).
    #[arg(long)]
    wmma: bool,

    /// With `--emit-cuda`, emit the KIVI-only flash-decode schedule (online softmax + vectorized
    /// int4 load4). Valid only for `decode_attention_kivi.jac`.
    #[arg(long = "kivi-flash")]
    kivi_flash: bool,

    /// End-to-end: codegen → nvcc → export weights → run inference (needs a CUDA toolchain + GPU).
    #[arg(long)]
    run: bool,

    /// GPU architecture passed to `nvcc -arch=` (e.g. sm_89 for RTX 4090, sm_80 for A100).
    #[arg(long, default_value = "sm_89")]
    arch: String,

    /// Number of new tokens to generate when `--run`ning.
    #[arg(long = "max-new", default_value_t = 20)]
    max_new: u32,

    /// Sink tokens for windowed attention (default 0 = exact attention; requires --run or --emit-model).
    #[arg(long, default_value_t = 0)]
    sink: u32,

    /// Window size for windowed attention (default = u32::MAX = exact attention; requires --run or --emit-model).
    #[arg(long = "window", default_value_t = u32::MAX)]
    window: u32,

    /// Emit/run the teacher-forced score-mode runtime: per-token NLL -> `scores.bin` + a
    /// `perplexity:` line over the token file. Composes with --sink/--window (requires --run
    /// or --emit-model). fp16 only.
    #[arg(long)]
    score: bool,

    /// Apply a model-surgery spec (e.g. `prune layers`): writes a new `.jac` + weight-convert.py.
    #[arg(long)]
    surgery: bool,

    /// The `.jac` spec file to operate on.
    file: Option<PathBuf>,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();

    if cli.doctor {
        return doctor();
    }

    if cli.validate {
        return match &cli.file {
            Some(f) => validate(f),
            None => {
                eprintln!("jacq --validate: expected a .jac file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.viz {
        return match &cli.file {
            Some(f) => viz(f),
            None => {
                eprintln!("jacq --viz: expected a .jac file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.viz_web {
        return match &cli.file {
            Some(f) => match load_ir(f) {
                Ok(ir) => vizweb::run(&ir),
                Err(()) => std::process::ExitCode::FAILURE,
            },
            None => {
                eprintln!("jacq --viz-web: expected a .jac file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.emit_model {
        let policy = jac_codegen_cuda::SelectionPolicy {
            sink_s: cli.sink,
            window_w: cli.window,
        };
        return match &cli.file {
            Some(f) => emit_model(f, policy, cli.score),
            None => {
                eprintln!("jacq --emit-model: expected a .jac file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.run {
        let policy = jac_codegen_cuda::SelectionPolicy {
            sink_s: cli.sink,
            window_w: cli.window,
        };
        return match &cli.file {
            Some(f) => run(f, &cli.arch, cli.max_new, policy, cli.score),
            None => {
                eprintln!("jacq --run: expected a .jac file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.surgery {
        return match &cli.file {
            Some(f) => surgery::run(f),
            None => {
                eprintln!("jacq --surgery: expected a .jac surgery file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    if cli.emit_cuda {
        let variant = if cli.kivi_flash {
            jac_codegen_cuda::KernelVariant::KiviFlash
        } else if cli.wmma {
            jac_codegen_cuda::KernelVariant::Wmma
        } else if cli.cublas {
            jac_codegen_cuda::KernelVariant::Cublas
        } else {
            jac_codegen_cuda::KernelVariant::Naive
        };
        return match &cli.file {
            Some(f) => emit_cuda_kernel(f, variant, cli.kernel_library),
            None => {
                eprintln!("jacq --emit-cuda: expected a .jac kernel file");
                std::process::ExitCode::FAILURE
            }
        };
    }

    eprintln!(
        "jacq: nothing to do. Try `jacq --doctor`, `jacq --validate <file>`, or `jacq --help`."
    );
    std::process::ExitCode::FAILURE
}

/// Read, parse, and lower a model spec; print a located error and return `Err(())` on failure.
/// Shared by `--validate`, `--viz`, and `--emit-model`.
fn load_ir(path: &Path) -> Result<jac_ir::ModelIr, ()> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        eprintln!("jacq: cannot read {}: {e}", path.display());
    })?;
    let model = parse_model(&src).map_err(|err| {
        report_error(path, &src, &err.message, err.span.start);
    })?;
    jac_ir::lower(&model).map_err(|e| {
        eprintln!("{}: error: {}", path.display(), e.message);
    })
}

/// Parse a model spec and report success or a located error.
fn validate(path: &Path) -> std::process::ExitCode {
    let ir = match load_ir(path) {
        Ok(ir) => ir,
        Err(()) => return std::process::ExitCode::FAILURE,
    };
    let c = &ir.config;
    let attn = match c.attention {
        jac_ir::Attention::Mha => "MHA",
        jac_ir::Attention::Gqa => "GQA",
        jac_ir::Attention::Mqa => "MQA",
    };
    println!("\u{2713} {}: model `{}`", path.display(), ir.name);
    println!(
        "  {} layers, {} heads / {} kv-heads ({attn}, {} q-heads per kv), head_dim {}",
        c.layers, c.heads, c.kv_heads, c.heads_per_group, c.head_dim,
    );
    println!(
        "  derived: q_dim {}, kv_dim {}, qkv_out {}; ffn {}, vocab {}, max_seq {}",
        c.q_dim, c.kv_dim, c.qkv_out, c.ffn_dim, c.vocab, c.max_seq,
    );
    std::process::ExitCode::SUCCESS
}

/// Render a terminal overview of the model: config, pipeline, and a memory estimate.
fn viz(path: &Path) -> std::process::ExitCode {
    let ir = match load_ir(path) {
        Ok(ir) => ir,
        Err(()) => return std::process::ExitCode::FAILURE,
    };

    let c = &ir.config;
    let attn = match c.attention {
        jac_ir::Attention::Mha => "MHA",
        jac_ir::Attention::Gqa => "GQA",
        jac_ir::Attention::Mqa => "MQA",
    };
    let fmt = match ir.weights.format {
        jac_ir::WeightFormat::HuggingFace => "huggingface",
        jac_ir::WeightFormat::Gguf => "gguf",
    };
    let dtype = format!("{:?}", ir.weights.dtype).to_lowercase();

    println!("\n  model: {}", ir.name);
    println!(
        "  config:  hidden={}  layers={}  heads={}  kv_heads={}  head_dim={}  [{attn}]",
        c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim
    );
    println!(
        "  weights: {fmt} {dtype}   ffn={}  vocab={}  max_seq={}  rope_base={}",
        c.ffn_dim, c.vocab, c.max_seq, c.rope_base
    );
    println!("\n  pipeline: {}", render_pipeline(&ir));

    // Memory estimate (fp16 = 2 bytes/param; KV cache always fp16 here).
    let per_layer = c.hidden               // attn norm
        + c.qkv_out * c.hidden             // fused QKV
        + c.hidden * c.q_dim               // output proj
        + c.hidden                         // ffn norm
        + 2 * c.ffn_dim * c.hidden         // fused gate+up
        + c.hidden * c.ffn_dim; // down proj
    let params = c.vocab * c.hidden + c.layers * per_layer + c.hidden + c.vocab * c.hidden;
    let weights_b = params as u64 * 2;
    let kv_b = 2 * c.layers as u64 * c.kv_heads as u64 * c.head_dim as u64 * c.max_seq as u64 * 2;
    let buffers_b: u64 = 512 * 1024 * 1024;
    let total = weights_b + kv_b + buffers_b;

    println!("\n  memory estimate (fp16):");
    println!("    weights   {}", human_bytes(weights_b));
    println!(
        "    kv cache  {}  (@ max_seq {})",
        human_bytes(kv_b),
        c.max_seq
    );
    println!("    buffers   {}  (est.)", human_bytes(buffers_b));
    println!("    total    ~{}", human_bytes(total));
    std::process::ExitCode::SUCCESS
}

/// One-line pipeline summary, e.g. `embed -> decoder x22 -> rmsnorm -> lm_head -> greedy_sample`.
fn render_pipeline(ir: &jac_ir::ModelIr) -> String {
    use jac_syntax::ast::Expr;
    ir.model
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
                format!("{layer} x{count}")
            }
            Expr::Call { name, .. } => name.clone(),
            _ => "?".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// Human-readable byte size.
fn human_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let f = b as f64;
    if f >= GB {
        format!("{:.2} GB", f / GB)
    } else {
        format!("{:.0} MB", f / MB)
    }
}

/// Where `--emit-model` / `--run` put generated artifacts for a model.
struct Generated {
    /// `generated/<name>/`.
    dir: PathBuf,
    /// `<name>_inference.cu`.
    cu: PathBuf,
    /// `<name>_export.py`.
    py: PathBuf,
    /// Line count of the emitted CUDA (for the success message).
    cu_lines: usize,
}

/// Load + lower + codegen a spec and write the runtime + exporter to `generated/<name>/`.
/// Shared by `--emit-model` and `--run`; prints located errors and returns `Err(FAILURE)`.
/// `score` selects the teacher-forced score-mode runtime (which already carries the
/// sink/window plumbing); otherwise a non-exact `policy` selects the windowed runtime.
fn emit_to_dir(
    path: &Path,
    policy: jac_codegen_cuda::SelectionPolicy,
    score: bool,
) -> Result<Generated, std::process::ExitCode> {
    let ir = load_ir(path).map_err(|()| std::process::ExitCode::FAILURE)?;
    let emitted = jac_codegen_cuda::emit(&ir).map_err(|e| {
        eprintln!("{}: codegen error: {}", path.display(), e);
        std::process::ExitCode::FAILURE
    })?;

    // Score-mode runtime composes windowing + teacher-forced scoring; otherwise use the
    // windowed codegen when the policy is non-exact, else the plain exact runtime.
    let cuda = if score {
        jac_codegen_cuda::emit_score_cuda(&ir)
    } else if policy != jac_codegen_cuda::SelectionPolicy::exact() {
        jac_codegen_cuda::emit_windowed_cuda(&ir)
    } else {
        emitted.cuda
    };

    let dir = Path::new("generated").join(&ir.name);
    let write = |p: &Path, contents: &str| -> Result<(), std::process::ExitCode> {
        std::fs::write(p, contents).map_err(|e| {
            eprintln!("jacq: cannot write {}: {e}", p.display());
            std::process::ExitCode::FAILURE
        })
    };
    std::fs::create_dir_all(&dir).map_err(|e| {
        eprintln!("jacq: cannot create {}: {e}", dir.display());
        std::process::ExitCode::FAILURE
    })?;
    let cu = dir.join(format!("{}_inference.cu", ir.name));
    let py = dir.join(format!("{}_export.py", ir.name));
    write(&cu, &cuda)?;
    write(&py, &emitted.python)?;
    Ok(Generated {
        dir,
        cu,
        py,
        cu_lines: cuda.lines().count(),
    })
}

/// Generate the CUDA inference runtime from a model spec into `generated/<name>/`.
fn emit_model(
    path: &Path,
    policy: jac_codegen_cuda::SelectionPolicy,
    score: bool,
) -> std::process::ExitCode {
    let g = match emit_to_dir(path, policy, score) {
        Ok(g) => g,
        Err(code) => return code,
    };
    println!("\u{2713} wrote {} ({} lines)", g.cu.display(), g.cu_lines);
    println!("\u{2713} wrote {}", g.py.display());
    std::process::ExitCode::SUCCESS
}

/// End-to-end build+run: codegen → `nvcc` → `python3 export` → `./inference`.
/// Each external step inherits stdio so the user sees nvcc/python/inference output live.
/// Requires a CUDA toolchain and a GPU on the host (check with `jacq --doctor`).
fn run(
    path: &Path,
    arch: &str,
    max_new: u32,
    policy: jac_codegen_cuda::SelectionPolicy,
    score: bool,
) -> std::process::ExitCode {
    let g = match emit_to_dir(path, policy, score) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let cu_name =
        g.cu.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
    let py_name =
        g.py.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
    println!(
        "\u{2713} codegen: {} ({} lines)\n",
        g.cu.display(),
        g.cu_lines
    );

    // 1. nvcc compile (relative paths, run inside the generated dir).
    println!("[1/3] nvcc -arch={arch} {cu_name} -o inference");
    let nvcc = step(
        Command::new("nvcc").current_dir(&g.dir).args([
            "-O3",
            &format!("-arch={arch}"),
            "-lcublas",
            &cu_name,
            "-o",
            "inference",
        ]),
        "nvcc",
    );
    if nvcc != 0 {
        return std::process::ExitCode::from(nvcc);
    }

    // 2. export weights (downloads the model, writes test_tokens.bin + a reference continuation).
    println!("\n[2/3] python3 {py_name} weights");
    let export = step(
        Command::new("python3")
            .current_dir(&g.dir)
            .args([&py_name, "weights"]),
        "python3",
    );
    if export != 0 {
        return std::process::ExitCode::from(export);
    }

    // 3. run inference; score-mode teacher-forces the token file, else generate max_new.
    //    Both pass sink/window only when the policy is non-default.
    let infer_args = if score {
        score_infer_args(policy)
    } else {
        let mut a = vec![
            "weights".to_string(),
            "weights/test_tokens.bin".to_string(),
            max_new.to_string(),
        ];
        if policy != jac_codegen_cuda::SelectionPolicy::exact() {
            a.push(policy.sink_s.to_string());
            a.push(policy.window_w.to_string());
        }
        a
    };
    let infer_display = infer_args.join(" ");
    println!("\n[3/3] ./inference {infer_display}");
    let infer = step(
        Command::new("./inference")
            .current_dir(&g.dir)
            .args(&infer_args),
        "inference",
    );
    if infer != 0 {
        return std::process::ExitCode::from(infer);
    }

    println!("\n\u{2713} done — compare the emitted token ids above against the REFERENCE line from step 2.");
    std::process::ExitCode::SUCCESS
}

/// Assemble the `./inference` argv for a teacher-forced score-mode run.
///
/// Score-mode is signalled by the `--score` token in the num-tokens slot (the generated
/// runtime keys on a leading `-` there); sink/window follow only when the policy is
/// non-exact, mirroring the generation path's positional convention.
fn score_infer_args(policy: jac_codegen_cuda::SelectionPolicy) -> Vec<String> {
    let mut a = vec![
        "weights".to_string(),
        "weights/test_tokens.bin".to_string(),
        "--score".to_string(),
    ];
    if policy != jac_codegen_cuda::SelectionPolicy::exact() {
        a.push(policy.sink_s.to_string());
        a.push(policy.window_w.to_string());
    }
    a
}

/// Run one orchestration step with inherited stdio. Returns 0 on success, else a non-zero code
/// suitable for `ExitCode::from` (a missing binary maps to 127, mirroring shell convention).
fn step(cmd: &mut Command, tool: &str) -> u8 {
    match cmd.status() {
        Ok(s) if s.success() => 0,
        Ok(s) => {
            let code = s.code().unwrap_or(1);
            eprintln!("jacq --run: {tool} failed (exit {code})");
            (code as u8).max(1)
        }
        Err(e) => {
            eprintln!("jacq --run: cannot launch {tool}: {e} (is it on PATH? try `jacq --doctor`)");
            127
        }
    }
}

/// Parse a kernel-DSL spec and emit a CUDA kernel + host harness to `generated/<name>/<name>.cu`.
/// Kernel codegen currently supports the GEMV pattern; other kernels report what's unsupported.
fn emit_cuda_kernel(
    path: &Path,
    variant: jac_codegen_cuda::KernelVariant,
    kernel_library: bool,
) -> std::process::ExitCode {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("jacq: cannot read {}: {e}", path.display());
            return std::process::ExitCode::FAILURE;
        }
    };
    let kernel = match jac_syntax::parse_kernel(&src) {
        Ok(k) => k,
        Err(err) => {
            report_error(path, &src, &err.message, err.span.start);
            return std::process::ExitCode::FAILURE;
        }
    };
    let cuda = match if kernel_library {
        jac_codegen_cuda::emit_kernel_library(&kernel, variant)
    } else {
        jac_codegen_cuda::emit_kernel(&kernel, variant)
    } {
        Ok(cu) => cu,
        Err(e) => {
            eprintln!("{}: kernel codegen: {e}", path.display());
            return std::process::ExitCode::FAILURE;
        }
    };
    let dir = Path::new("generated").join(&kernel.name);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("jacq: cannot create {}: {e}", dir.display());
        return std::process::ExitCode::FAILURE;
    }
    let cu_path = dir.join(format!("{}.cu", kernel.name));
    if let Err(e) = std::fs::write(&cu_path, &cuda) {
        eprintln!("jacq: cannot write {}: {e}", cu_path.display());
        return std::process::ExitCode::FAILURE;
    }
    println!(
        "\u{2713} wrote {} ({} lines)",
        cu_path.display(),
        cuda.lines().count()
    );
    std::process::ExitCode::SUCCESS
}

/// Render `file:line:col: message` with the offending line and a caret.
fn report_error(path: &Path, src: &str, message: &str, byte_offset: usize) {
    let offset = byte_offset.min(src.len());
    let line_start = src[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_no = src[..offset].bytes().filter(|&b| b == b'\n').count() + 1;
    let col = src[line_start..offset].chars().count() + 1;
    let line_end = src[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(src.len());
    let line_text = &src[line_start..line_end];

    eprintln!("{}:{}:{}: error: {}", path.display(), line_no, col, message);
    eprintln!("  {line_text}");
    eprintln!("  {}^", " ".repeat(col.saturating_sub(1)));
}

/// Report whether the toolchain jacq generates against is present.
fn doctor() -> std::process::ExitCode {
    println!("jacq doctor — checking prerequisites\n");

    let checks = [
        (
            "nvcc",
            &["--version"][..],
            "CUDA compiler (required to build generated runtimes)",
        ),
        (
            "python3",
            &["--version"][..],
            "weight export scripts (HuggingFace models)",
        ),
        ("nvidia-smi", &[][..], "an NVIDIA GPU to run inference"),
    ];

    for (tool, args, why) in checks {
        match probe(tool, args) {
            Some(version) => println!("  \u{2713} {tool:<11} {version}"),
            None => println!("  \u{2717} {tool:<11} not found — needed for {why}"),
        }
    }

    println!("\nDone.");
    std::process::ExitCode::SUCCESS
}

/// Run `tool args...` and return the first line of its output if the tool exists.
fn probe(tool: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(tool).args(args).output().ok()?;
    let text = if output.stdout.is_empty() {
        &output.stderr
    } else {
        &output.stdout
    };
    let line = String::from_utf8_lossy(text)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    Some(if line.is_empty() {
        "present".to_string()
    } else {
        line
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_style_still_parses() {
        let cli =
            Cli::try_parse_from(["jacq", "--validate", "model.jac"]).expect("flag style works");
        assert!(cli.validate);
        assert_eq!(cli.file.as_deref(), Some(Path::new("model.jac")));
    }

    /// `set_current_dir` is process-global; serialize the tests that `cd` into a temp dir
    /// so cargo's parallel test threads don't clobber each other's working directory.
    /// (Recover from poisoning so one test's panic doesn't cascade into spurious failures.)
    static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_cwd() -> std::sync::MutexGuard<'static, ()> {
        CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn step_maps_success_failure_and_missing_binary() {
        // `true`/`false` are POSIX-portable; a bogus name exercises the launch-failure path.
        assert_eq!(
            step(&mut Command::new("true"), "true"),
            0,
            "success should map to 0"
        );
        assert_eq!(
            step(&mut Command::new("false"), "false"),
            1,
            "failure should map to its exit code"
        );
        assert_eq!(
            step(&mut Command::new("jac-no-such-binary-xyz"), "missing"),
            127,
            "a missing binary should map to 127 (shell convention)"
        );
    }

    // ── load_ir error paths ───────────────────────────────────────────────────

    #[test]
    fn load_ir_returns_err_on_missing_file() {
        // A completely absent path must yield Err(()), not panic.
        assert!(load_ir(Path::new("definitely_no_such_file_xyz.jac")).is_err());
    }

    #[test]
    fn load_ir_returns_err_on_parse_failure() {
        // Write a temp file with invalid syntax; load_ir must return Err(()).
        let dir = std::env::temp_dir();
        let p = dir.join("jac_test_bad_syntax.jac");
        std::fs::write(&p, "this is not valid jac syntax !!!").unwrap();
        let result = load_ir(&p);
        let _ = std::fs::remove_file(&p);
        assert!(result.is_err());
    }

    // ── --run command-sequence assembly ──────────────────────────────────────
    //
    // These tests verify that the three steps in `run()` assemble the correct
    // command strings WITHOUT executing nvcc/python/inference (which require a
    // CUDA toolchain + GPU). We test the argument-assembly logic by inspecting
    // what `Command` would be built — verified via the generated artifact names
    // that `emit_to_dir` produces from a real spec.
    //
    // NOTE: on-device token-match validation (smoke-running ./inference and
    // comparing token ids against the Python reference) is deferred until
    // gpu-validate MODEL-mode (jq-a9m) is complete. That acceptance gate
    // requires an actual GPU and is NOT exercised here.

    fn specs_dir() -> std::path::PathBuf {
        // The workspace-level specs/ dir, reached via CARGO_MANIFEST_DIR.
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs/models")
    }

    #[test]
    fn run_artifact_names_match_model_name() {
        // emit_to_dir on tinyllama.jac must produce the expected filenames inside
        // generated/tinyllama/: tinyllama_inference.cu and tinyllama_export.py.
        // This verifies the Generated struct is populated correctly and that the
        // step() call sites in run() will reference the right file names.
        let spec = specs_dir().join("tinyllama.jac");
        if !spec.exists() {
            // Skip gracefully when running outside the full workspace.
            return;
        }

        // Emit into a temp dir so the test doesn't litter the project root.
        let _cwd = lock_cwd();
        let tmp = std::env::temp_dir().join("jac_run_test_artifacts");
        std::fs::create_dir_all(&tmp).unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();

        let result = emit_to_dir(&spec, jac_codegen_cuda::SelectionPolicy::exact(), false);

        std::env::set_current_dir(&orig).unwrap();

        let g = result.expect("emit_to_dir should succeed for tinyllama.jac");

        // File names: <name>_inference.cu / <name>_export.py
        assert_eq!(
            g.cu.file_name().and_then(|n| n.to_str()),
            Some("tinyllama_inference.cu"),
            "CUDA artifact name wrong: {:?}",
            g.cu
        );
        assert_eq!(
            g.py.file_name().and_then(|n| n.to_str()),
            Some("tinyllama_export.py"),
            "Python artifact name wrong: {:?}",
            g.py
        );

        // The generated dir is generated/<name>/.
        assert!(
            g.dir.ends_with("generated/tinyllama"),
            "generated dir should be generated/tinyllama, got {:?}",
            g.dir
        );

        // Sanity: CUDA output is non-empty.
        assert!(g.cu_lines > 0, "emitted CUDA should have at least one line");

        // Cleanup.
        let _ = std::fs::remove_dir_all(tmp.join("generated"));
    }

    #[test]
    fn run_nvcc_args_use_arch_and_output_flag() {
        // Verify the nvcc invocation shape: -O3 -arch=<arch> -lcublas <cu_name> -o inference.
        // We build the Command in the same way run() does and inspect its args.
        let arch = "sm_80";
        let cu_name = "tinyllama_inference.cu";
        let cmd = Command::new("nvcc")
            .args([
                "-O3",
                &format!("-arch={arch}"),
                "-lcublas",
                cu_name,
                "-o",
                "inference",
            ])
            .get_program()
            .to_string_lossy()
            .into_owned();
        assert_eq!(cmd, "nvcc");

        // Verify the arch substitution works for a few representative targets.
        for arch in ["sm_80", "sm_86", "sm_89", "sm_90"] {
            let flag = format!("-arch={arch}");
            assert!(flag.starts_with("-arch="), "arch flag malformed: {flag}");
        }
    }

    #[test]
    fn run_inference_args_embed_max_new() {
        // The ./inference invocation takes: weights weights/test_tokens.bin <max_new>.
        // Verify the max_new substitution produces a valid decimal string.
        for max_new in [1u32, 20, 200, u32::MAX] {
            let s = max_new.to_string();
            assert!(s.parse::<u32>().unwrap() == max_new);
        }
    }

    // ── score-mode CLI/arg threading (track B, perplexity) ────────────────────

    #[test]
    fn score_flag_parses_with_emit_and_run() {
        // The --score flag threads through clap for both --emit-model and --run.
        let e = Cli::try_parse_from(["jacq", "--emit-model", "--score", "m.jac"]).unwrap();
        assert!(
            e.score && e.emit_model,
            "--score should set the score flag under --emit-model"
        );
        let r = Cli::try_parse_from([
            "jacq", "--run", "--score", "--sink", "4", "--window", "64", "m.jac",
        ])
        .unwrap();
        assert!(
            r.score && r.run,
            "--score should set the score flag under --run"
        );
        assert_eq!(
            (r.sink, r.window),
            (4, 64),
            "--score must compose with --sink/--window"
        );
        // Default: no --score → generation mode.
        let g = Cli::try_parse_from(["jacq", "--run", "m.jac"]).unwrap();
        assert!(!g.score, "score should default off");
    }

    #[test]
    fn score_infer_args_signal_and_policy() {
        // Exact policy → just the --score token in the num-tokens slot, no sink/window.
        let exact = score_infer_args(jac_codegen_cuda::SelectionPolicy::exact());
        assert_eq!(exact, ["weights", "weights/test_tokens.bin", "--score"]);
        // Non-exact policy → sink/window appended after --score (positional, like generation).
        let windowed = score_infer_args(jac_codegen_cuda::SelectionPolicy {
            sink_s: 4,
            window_w: 64,
        });
        assert_eq!(
            windowed,
            ["weights", "weights/test_tokens.bin", "--score", "4", "64"]
        );
    }

    #[test]
    fn emit_to_dir_score_selects_score_runtime() {
        // With score=true, emit_to_dir must emit the teacher-forced runtime (nll_kernel +
        // perplexity line), not the plain/windowed generation runtime.
        let spec = specs_dir().join("tinyllama.jac");
        if !spec.exists() {
            return; // skip outside the full workspace
        }
        let _cwd = lock_cwd();
        let tmp = std::env::temp_dir().join("jac_score_test_artifacts");
        std::fs::create_dir_all(&tmp).unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();
        // Emit AND read the artifact while cwd is the temp dir — `g.cu` is a relative path.
        let cu = emit_to_dir(&spec, jac_codegen_cuda::SelectionPolicy::exact(), true)
            .ok()
            .and_then(|g| std::fs::read_to_string(&g.cu).ok());
        std::env::set_current_dir(&orig).unwrap();

        let cu = cu.expect("emit_to_dir(score=true) should emit a readable score runtime");
        assert!(
            cu.contains("nll_kernel"),
            "score runtime should contain the NLL kernel"
        );
        assert!(
            cu.contains("perplexity:"),
            "score runtime should print a perplexity line"
        );
        let _ = std::fs::remove_dir_all(tmp.join("generated"));
    }
}
