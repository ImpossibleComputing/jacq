//! Golden-snapshot tests for the kernel-DSL CUDA codegen.
//!
//! Each committed kernel spec is emitted in every supported variant and compared byte-for-byte
//! against a reviewed reference under `tests/snapshots/kernels/`. These lock in the emitted
//! kernel output (gemv/matmul × naive/cuBLAS) so a refactor can't silently change a kernel. The
//! parallel of `golden.rs` (which covers the model runtimes).
//!
//! Regenerate after an intentional change:
//!   UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden_kernels

use std::path::PathBuf;

use jac_codegen_cuda::{emit_subject_harness, KernelVariant};
use jac_syntax::SubjectSrc;

fn specs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs/kernels")
}

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/kernels")
}

/// Emit `spec` in `variant` and assert it matches `tests/snapshots/kernels/<snapshot>`
/// (or rewrite it when `UPDATE_SNAPSHOTS` is set).
fn check(spec: &str, variant: KernelVariant, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let kernel = jac_syntax::parse_kernel(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let cuda = jac_codegen_cuda::emit_kernel(&kernel, variant)
        .unwrap_or_else(|e| panic!("{spec} ({variant:?}): codegen error: {e}"));

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden_kernels`",
            snap_path.display()
        )
    });
    if cuda != expected {
        let (g, e): (Vec<&str>, Vec<&str>) = (cuda.lines().collect(), expected.lines().collect());
        let i = (0..g.len().min(e.len()))
            .find(|&i| g[i] != e[i])
            .unwrap_or(g.len().min(e.len()));
        panic!(
            "generated CUDA for `{spec}` ({variant:?}) differs from `{snapshot}` at line {}:\n  \
             expected: {}\n  got:      {}\nregenerate with UPDATE_SNAPSHOTS=1 if intentional.",
            i + 1,
            e.get(i).copied().unwrap_or("<eof>"),
            g.get(i).copied().unwrap_or("<eof>"),
        );
    }
}

#[test]
fn snapshot_gemv_naive() {
    check("gemv.jac", KernelVariant::Naive, "gemv_naive.cu");
}

#[test]
fn snapshot_gemv_cublas() {
    check("gemv.jac", KernelVariant::Cublas, "gemv_cublas.cu");
}

#[test]
fn snapshot_matmul_naive() {
    check("matmul.jac", KernelVariant::Naive, "matmul_naive.cu");
}

#[test]
fn snapshot_matmul_cublas() {
    check("matmul.jac", KernelVariant::Cublas, "matmul_cublas.cu");
}

#[test]
fn snapshot_softmax() {
    // softmax has a single (naive) backend; cuBLAS doesn't apply.
    check("softmax.jac", KernelVariant::Naive, "softmax.cu");
}

#[test]
fn snapshot_matmul_wmma() {
    // WMMA tensor-core variant; matmul.jac has M=256/N=128/K=64 — all multiples of 16.
    check("matmul.jac", KernelVariant::Wmma, "matmul_wmma.cu");
}

/// Emit `spec` via [`jac_codegen_cuda::emit_kernel_library`] in `variant` and assert it matches
/// `tests/snapshots/kernels/<snapshot>` (or rewrite it when `UPDATE_SNAPSHOTS` is set).
///
/// Unlike `check`, this exercises the runtime-extension library-artifact path used by source-vLLM
/// serving lanes (no standalone `main()`/file-IO harness) — the actual artifact a served kernel
/// (e.g. the KIVI paged decode lane) ships. `check` alone does not protect this path.
fn check_library(spec: &str, variant: KernelVariant, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let src = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let kernel = jac_syntax::parse_kernel(&src)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let cuda = jac_codegen_cuda::emit_kernel_library(&kernel, variant)
        .unwrap_or_else(|e| panic!("{spec} ({variant:?}): library codegen error: {e}"));

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden_kernels`",
            snap_path.display()
        )
    });
    if cuda != expected {
        let (g, e): (Vec<&str>, Vec<&str>) = (cuda.lines().collect(), expected.lines().collect());
        let i = (0..g.len().min(e.len()))
            .find(|&i| g[i] != e[i])
            .unwrap_or(g.len().min(e.len()));
        panic!(
            "generated library CUDA for `{spec}` ({variant:?}) differs from `{snapshot}` at line {}:\n  \
             expected: {}\n  got:      {}\nregenerate with UPDATE_SNAPSHOTS=1 if intentional.",
            i + 1,
            e.get(i).copied().unwrap_or("<eof>"),
            g.get(i).copied().unwrap_or("<eof>"),
        );
    }
}

/// Check (or regenerate) the benchmark harness for gemv/naive.
fn check_harness(spec: &str, src: SubjectSrc, snapshot: &str) {
    let spec_path = specs_dir().join(spec);
    let raw = std::fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", spec_path.display()));
    let kernel = jac_syntax::parse_kernel(&raw)
        .unwrap_or_else(|e| panic!("{spec}: parse error: {}", e.message));
    let cuda = emit_subject_harness(&kernel, &src)
        .unwrap_or_else(|e| panic!("{spec} harness codegen error: {e}"));

    let snap_path = snapshots_dir().join(snapshot);
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        std::fs::create_dir_all(snapshots_dir()).unwrap();
        std::fs::write(&snap_path, &cuda).unwrap();
        eprintln!("updated snapshot {}", snap_path.display());
        return;
    }

    let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|e| {
        panic!(
            "missing snapshot {} ({e}); regenerate with `UPDATE_SNAPSHOTS=1 cargo test -p jac-codegen-cuda --test golden_kernels`",
            snap_path.display()
        )
    });
    if cuda != expected {
        let (g, e): (Vec<&str>, Vec<&str>) = (cuda.lines().collect(), expected.lines().collect());
        let i = (0..g.len().min(e.len()))
            .find(|&i| g[i] != e[i])
            .unwrap_or(g.len().min(e.len()));
        panic!(
            "harness for `{spec}` differs from `{snapshot}` at line {}:\n  \
             expected: {}\n  got:      {}\nregenerate with UPDATE_SNAPSHOTS=1 if intentional.",
            i + 1,
            e.get(i).copied().unwrap_or("<eof>"),
            g.get(i).copied().unwrap_or("<eof>"),
        );
    }
}

#[test]
fn snapshot_gemv_harness_naive() {
    check_harness(
        "gemv.jac",
        SubjectSrc::Jac("naive".into()),
        "gemv_harness_naive.cu",
    );
}

#[test]
fn snapshot_attention_naive() {
    // jac-native decode attention, standalone .cu (the `jacq --emit-cuda` path). EXACT base variant.
    check(
        "decode_attention.jac",
        KernelVariant::Naive,
        "decode_attention_naive.cu",
    );
}

#[test]
fn snapshot_attention_harness_naive() {
    // The experiment-harness wrapper around the jac-generated attention body (jac("naive") subject).
    check_harness(
        "decode_attention.jac",
        SubjectSrc::Jac("naive".into()),
        "decode_attention_harness_naive.cu",
    );
}

#[test]
fn snapshot_attention_windowed_naive() {
    // Windowed variant (sink+window mask via SINK/WINDOW params) — generates the masked.cu equivalent.
    check(
        "decode_attention_windowed.jac",
        KernelVariant::Naive,
        "decode_attention_windowed_naive.cu",
    );
}

#[test]
fn snapshot_attention_windowed_harness_naive() {
    check_harness(
        "decode_attention_windowed.jac",
        SubjectSrc::Jac("naive".into()),
        "decode_attention_windowed_harness_naive.cu",
    );
}

#[test]
fn snapshot_kivi_attention_naive() {
    // First KIVI/KV-quant DSL lane: fused int4 decode over quantized old KV + fp16 residual.
    check(
        "decode_attention_kivi.jac",
        KernelVariant::Naive,
        "decode_attention_kivi_naive.cu",
    );
}

#[test]
fn snapshot_kivi_attention_harness_naive() {
    check_harness(
        "decode_attention_kivi.jac",
        SubjectSrc::Jac("naive".into()),
        "decode_attention_kivi_harness_naive.cu",
    );
}

#[test]
fn snapshot_kivi_attention_flash() {
    // Single-split flash schedule: online softmax + vectorized int4 load4.
    check(
        "decode_attention_kivi.jac",
        KernelVariant::KiviFlash,
        "decode_attention_kivi_flash.cu",
    );
}

#[test]
fn snapshot_kivi_attention_harness_flash() {
    check_harness(
        "decode_attention_kivi.jac",
        SubjectSrc::Jac("kivi_flash".into()),
        "decode_attention_kivi_harness_flash.cu",
    );
}

#[test]
fn snapshot_kivi_paged_attention_library_flash() {
    // Source-vLLM served paged decode: BLOCK_TABLE[logical_page] -> physical page indirection
    // over K/V_CODE_PAGES etc. This is the emit_kernel_library artifact the KIVI vLLM serving
    // lane actually ships (integrations/vllm_kivi) -- previously had zero Rust-level snapshot
    // coverage, unlike every other served/benchmarked kernel shape in this file.
    check_library(
        "decode_attention_kivi_paged.jac",
        KernelVariant::KiviFlash,
        "decode_attention_kivi_paged_library_flash.cu",
    );
}

#[test]
fn snapshot_kivi_sparse_retained_attention_library_flash() {
    // Sparse-retained KIVI decode: compact retained physical-page table + NUM_RETAINED_PAGES
    // bounds the old-token loop. This is source/CPU readiness only until GPU bitcheck/goodput.
    check_library(
        "decode_attention_kivi_sparse_retained.jac",
        KernelVariant::KiviFlash,
        "decode_attention_kivi_sparse_retained_library_flash.cu",
    );
}

#[test]
fn snapshot_plain_paged_attention_library_flash() {
    // Plain (non-KIVI) paged GQA decode: the second real consumer of
    // emit_gathered_page_step, proving the gather-indexed codegen extraction
    // generalizes beyond KIVI. NH=28/NKV=4 -> GQA group_size 7, no int4/scale/zero/residual.
    check_library(
        "decode_attention_paged.jac",
        KernelVariant::PagedFlash,
        "decode_attention_paged_library_flash.cu",
    );
}

#[test]
fn snapshot_kivi_paged_batched_attention_library_flash() {
    // Batched sibling of the single-token paged-decode kernel above: adds SEQ_LENS and a
    // MAX_QUERIES-leading dim on Q/O so one launch covers a batch of decode queries instead of
    // one. variants.rs flags this kernel's parity with the single-token kernel as an
    // expectation that still requires GPU validation -- this snapshot is the first thing
    // pinning its emitted CUDA at all, and is what protects it (and the single-token kernel
    // above) across refactors.
    check_library(
        "decode_attention_kivi_paged_batched.jac",
        KernelVariant::KiviFlash,
        "decode_attention_kivi_paged_batched_library_flash.cu",
    );
}
