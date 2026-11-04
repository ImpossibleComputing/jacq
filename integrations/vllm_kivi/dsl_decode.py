"""DSL-generated KIVI decode kernel artifact wiring for the vLLM source lane.

This module does not compile or register a vLLM backend by itself.  It makes the
source-of-truth explicit for the next backend patch: the int4 KIVI decode CUDA
must be generated from `specs/kernels/decode_attention_kivi_paged.jac` with the
`kivi_flash` lowering, rather than hand-written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys

DSL_DECODE_VARIANT = "kivi_flash"
DSL_DECODE_SOURCE = "dsl-kivi-flash"
DEFAULT_SPEC = Path("specs/kernels/decode_attention_kivi_paged.jac")
GENERATED_REL = Path("generated/decode_attention_kivi_paged/decode_attention_kivi_paged.cu")
VLLM_KERNEL_REL = Path("csrc/kivi/decode_attention_kivi_dsl.cu")
TP2_LOCAL_VLLM_KERNEL_REL = Path("csrc/kivi/decode_attention_kivi_tp2_local_dsl.cu")


@dataclass(frozen=True)
class DslDecodeKernelArtifact:
    repo_root: str
    vllm_root: str
    spec: str = str(DEFAULT_SPEC)
    variant: str = DSL_DECODE_VARIANT
    source_label: str = DSL_DECODE_SOURCE
    generated_rel: str = str(GENERATED_REL)
    vllm_kernel_rel: str = str(VLLM_KERNEL_REL)

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def vllm_path(self) -> Path:
        return Path(self.vllm_root)

    @property
    def spec_path(self) -> Path:
        return self.repo_path / self.spec

    @property
    def generated_path(self) -> Path:
        return self.repo_path / self.generated_rel

    @property
    def vllm_kernel_path(self) -> Path:
        return self.vllm_path / self.vllm_kernel_rel

    def jac_emit_command(self) -> list[str]:
        return [
            "cargo",
            "run",
            "-q",
            "-p",
            "jac-cli",
            "--",
            "--emit-cuda",
            "--kernel-library",
            "--kivi-flash",
            self.spec,
        ]

    def report(self) -> dict[str, object]:
        return {
            **asdict(self),
            "spec_exists": self.spec_path.exists(),
            "generated_exists": self.generated_path.exists(),
            "vllm_kernel_exists": self.vllm_kernel_path.exists(),
            "jac_emit_command": self.jac_emit_command(),
            "serving_env": {
                "JAC_VLLM_KIVI_DECODE_KERNEL_SOURCE": self.source_label,
                "JAC_VLLM_KIVI_DECODE_KERNEL_CU": str(self.vllm_kernel_path),
            },
            "integrity_boundary": (
                "This only materializes the DSL-lowered CUDA source for the vLLM fork. "
                "It is not a CUDA bitcheck, not a paged/block-table backend, and not a served result."
            ),
        }

    def emit(self) -> Path:
        if self.variant != DSL_DECODE_VARIANT:
            raise ValueError(f"only {DSL_DECODE_VARIANT!r} is supported, got {self.variant!r}")
        if not self.spec_path.exists():
            raise FileNotFoundError(f"missing KIVI DSL spec: {self.spec_path}")
        try:
            proc = subprocess.run(
                self.jac_emit_command(),
                cwd=self.repo_path,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
            raise
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if not self.generated_path.exists():
            raise FileNotFoundError(
                f"jacq CLI completed but did not write expected CUDA: {self.generated_path}"
            )
        text = self.generated_path.read_text()
        required = (
            "Library artifact for embedding in a runtime extension",
            "extern \"C\" void jac_launch(",
            "source-vLLM paged KIVI4 flash decode",
            "static constexpr int NH = 40;",
            "static constexpr int NKV = 8;",
            "const int* BLOCK_TABLE",
            "block_table[logical_page]",
            "<<<NH, 32, 0, stream>>>",
        )
        missing = [needle for needle in required if needle not in text]
        forbidden = ("int main(", "load_half(", "load_u8(")
        present_forbidden = [needle for needle in forbidden if needle in text]
        if missing or present_forbidden:
            raise RuntimeError(
                "DSL KIVI decode artifact is not a vLLM-safe library translation unit; "
                f"missing={missing} forbidden={present_forbidden}"
            )
        self.vllm_kernel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.generated_path, self.vllm_kernel_path)
        return self.vllm_kernel_path


BATCHED_SPEC = Path("specs/kernels/decode_attention_kivi_paged_batched.jac")
BATCHED_GENERATED_REL = Path(
    "generated/decode_attention_kivi_paged_batched/decode_attention_kivi_paged_batched.cu"
)
BATCHED_VLLM_KERNEL_REL = Path("csrc/kivi/decode_attention_kivi_batched_dsl.cu")
TP2_LOCAL_BATCHED_VLLM_KERNEL_REL = Path(
    "csrc/kivi/decode_attention_kivi_batched_tp2_local_dsl.cu"
)


@dataclass(frozen=True)
class DslDecodeBatchedKernelArtifact:
    """Batched sibling of DslDecodeKernelArtifact (batched prefill-loop fix).

    Separate destination file (decode_attention_kivi_batched_dsl.cu, not
    decode_attention_kivi_dsl.cu) because both translation units are compiled into the same
    extension -- the batched kernel is an opt-in alternative to the single-token path, not a
    replacement for it (see jac_launch_batched's own naming for the matching link-time reason).
    """

    repo_root: str
    vllm_root: str
    spec: str = str(BATCHED_SPEC)
    variant: str = DSL_DECODE_VARIANT
    source_label: str = "dsl-kivi-flash-batched"
    generated_rel: str = str(BATCHED_GENERATED_REL)
    vllm_kernel_rel: str = str(BATCHED_VLLM_KERNEL_REL)

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def vllm_path(self) -> Path:
        return Path(self.vllm_root)

    @property
    def spec_path(self) -> Path:
        return self.repo_path / self.spec

    @property
    def generated_path(self) -> Path:
        return self.repo_path / self.generated_rel

    @property
    def vllm_kernel_path(self) -> Path:
        return self.vllm_path / self.vllm_kernel_rel

    def jac_emit_command(self) -> list[str]:
        return [
            "cargo", "run", "-q", "-p", "jac-cli", "--",
            "--emit-cuda", "--kernel-library", "--kivi-flash", self.spec,
        ]

    def report(self) -> dict[str, object]:
        return {
            **asdict(self),
            "spec_exists": self.spec_path.exists(),
            "generated_exists": self.generated_path.exists(),
            "vllm_kernel_exists": self.vllm_kernel_path.exists(),
            "jac_emit_command": self.jac_emit_command(),
            "serving_env": {
                "JAC_VLLM_KIVI_DECODE_BATCHED_KERNEL_SOURCE": self.source_label,
                "JAC_VLLM_KIVI_DECODE_BATCHED_KERNEL_CU": str(self.vllm_kernel_path),
            },
            "integrity_boundary": (
                "This only materializes the DSL-lowered CUDA source for the vLLM fork. "
                "It is not a CUDA bitcheck, not wired into the request loop, and not a served result."
            ),
        }

    def emit(self) -> Path:
        if self.variant != DSL_DECODE_VARIANT:
            raise ValueError(f"only {DSL_DECODE_VARIANT!r} is supported, got {self.variant!r}")
        if not self.spec_path.exists():
            raise FileNotFoundError(f"missing batched KIVI DSL spec: {self.spec_path}")
        proc = subprocess.run(
            self.jac_emit_command(),
            cwd=self.repo_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if not self.generated_path.exists():
            raise FileNotFoundError(
                f"jacq CLI completed but did not write expected CUDA: {self.generated_path}"
            )
        text = self.generated_path.read_text()
        required = (
            "Library artifact for embedding in a runtime extension",
            "extern \"C\" void jac_launch_batched(",
            "source-vLLM BATCHED paged KIVI4 flash decode",
            "static constexpr int NH = 40;",
            "static constexpr int NKV = 8;",
            "static constexpr int MAX_QUERIES = 160;",
            "const int* SEQ_LENS",
            "const int* BLOCK_TABLE",
            "block_table[logical_page]",
            "<<<dim3(NH, NUM_QUERIES), 32, 0, stream>>>",
        )
        missing = [needle for needle in required if needle not in text]
        forbidden = ("int main(", "load_half(", "load_u8(", "extern \"C\" void jac_launch(")
        present_forbidden = [needle for needle in forbidden if needle in text]
        if missing or present_forbidden:
            raise RuntimeError(
                "DSL batched KIVI decode artifact is not a vLLM-safe library translation unit; "
                f"missing={missing} forbidden={present_forbidden}"
            )
        self.vllm_kernel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.generated_path, self.vllm_kernel_path)
        return self.vllm_kernel_path


SPARSE_RETAINED_SPEC = Path("specs/kernels/decode_attention_kivi_sparse_retained.jac")
SPARSE_RETAINED_GENERATED_REL = Path(
    "generated/decode_attention_kivi_sparse_retained/decode_attention_kivi_sparse_retained.cu"
)
SPARSE_RETAINED_VLLM_KERNEL_REL = Path("csrc/kivi/decode_attention_kivi_sparse_retained_dsl.cu")
TP2_LOCAL_SPARSE_RETAINED_VLLM_KERNEL_REL = Path(
    "csrc/kivi/decode_attention_kivi_sparse_retained_tp2_local_dsl.cu"
)


def _materialize_tp2_local_variant(src: Path, dst: Path, *, symbol: str) -> Path:
    """Clone a global-Qwen32B artifact into the TP=2 local per-rank geometry.

    Served TP=2 deployments pass rank-local geometry into each worker:
    each rank sees local attention geometry ``NH=20/NKV=4``, not the global
    Qwen32B geometry. The DSL specs remain the global source of truth, but the
    source-build path also needs a same-codegen local variant with distinct C
    symbols so both global TP=1 and local TP=2 artifacts can link into vLLM._C.

    This is a source-materialization step, not a GPU correctness claim; the
    generated local artifacts still need the normal CUDA bitcheck/HTTP gates.
    """
    text = src.read_text()
    replacements = {
        "static constexpr int NH = 40;": "static constexpr int NH = 20;",
        "static constexpr int NKV = 8;": "static constexpr int NKV = 4;",
        "Qwen2.5-Coder-32B": "Qwen2.5-Coder-32B TP=2 local-rank",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if symbol == "single":
        text = text.replace('extern "C" void jac_launch(', 'extern "C" void jac_launch_tp2_local(')
        text = text.replace(
            "decode_attention_kivi_paged_kivi_flash_paged_decode",
            "decode_attention_kivi_paged_tp2_local_kivi_flash_paged_decode",
        )
    elif symbol == "batched":
        text = text.replace(
            'extern "C" void jac_launch_batched(',
            'extern "C" void jac_launch_batched_tp2_local(',
        )
        text = text.replace(
            "decode_attention_kivi_paged_batched_kivi_flash_paged_decode_batched",
            "decode_attention_kivi_paged_batched_tp2_local_kivi_flash_paged_decode_batched",
        )
    elif symbol == "sparse_retained":
        text = text.replace(
            'extern "C" void jac_launch_sparse_retained(',
            'extern "C" void jac_launch_sparse_retained_tp2_local(',
        )
        text = text.replace(
            "decode_attention_kivi_sparse_retained_kivi_flash_sparse_retained_decode",
            "decode_attention_kivi_sparse_retained_tp2_local_kivi_flash_sparse_retained_decode",
        )
    else:
        raise ValueError(f"unknown TP2-local symbol kind: {symbol}")
    required = ("static constexpr int NH = 20;", "static constexpr int NKV = 4;", "_tp2_local(")
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise RuntimeError(f"failed to materialize TP2-local KIVI artifact; missing={missing}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text)
    return dst


@dataclass(frozen=True)
class DslDecodeSparseRetainedKernelArtifact:
    """Sparse-retained decode sibling for page-aware KIVI eviction.

    This materializes a separate CUDA translation unit; it does not replace the dense decode
    op. Runtime wiring must pass a compact retained physical-page table plus NUM_RETAINED_PAGES,
    and GPU bitcheck/goodput must pass before served use.
    """

    repo_root: str
    vllm_root: str
    spec: str = str(SPARSE_RETAINED_SPEC)
    variant: str = DSL_DECODE_VARIANT
    source_label: str = "dsl-kivi-flash-sparse-retained"
    generated_rel: str = str(SPARSE_RETAINED_GENERATED_REL)
    vllm_kernel_rel: str = str(SPARSE_RETAINED_VLLM_KERNEL_REL)

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def vllm_path(self) -> Path:
        return Path(self.vllm_root)

    @property
    def spec_path(self) -> Path:
        return self.repo_path / self.spec

    @property
    def generated_path(self) -> Path:
        return self.repo_path / self.generated_rel

    @property
    def vllm_kernel_path(self) -> Path:
        return self.vllm_path / self.vllm_kernel_rel

    def jac_emit_command(self) -> list[str]:
        return [
            "cargo", "run", "-q", "-p", "jac-cli", "--",
            "--emit-cuda", "--kernel-library", "--kivi-flash", self.spec,
        ]

    def report(self) -> dict[str, object]:
        return {
            **asdict(self),
            "spec_exists": self.spec_path.exists(),
            "generated_exists": self.generated_path.exists(),
            "vllm_kernel_exists": self.vllm_kernel_path.exists(),
            "jac_emit_command": self.jac_emit_command(),
            "serving_env": {
                "JAC_VLLM_KIVI_DECODE_SPARSE_RETAINED_KERNEL_SOURCE": self.source_label,
                "JAC_VLLM_KIVI_DECODE_SPARSE_RETAINED_KERNEL_CU": str(self.vllm_kernel_path),
            },
            "integrity_boundary": (
                "This only materializes the DSL-lowered sparse-retained CUDA source. It is not a "
                "CUDA bitcheck, not wired into the request loop, and not a served-goodput claim."
            ),
        }

    def emit(self) -> Path:
        if self.variant != DSL_DECODE_VARIANT:
            raise ValueError(f"only {DSL_DECODE_VARIANT!r} is supported, got {self.variant!r}")
        if not self.spec_path.exists():
            raise FileNotFoundError(f"missing sparse-retained KIVI DSL spec: {self.spec_path}")
        proc = subprocess.run(
            self.jac_emit_command(),
            cwd=self.repo_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if not self.generated_path.exists():
            raise FileNotFoundError(
                f"jacq CLI completed but did not write expected CUDA: {self.generated_path}"
            )
        text = self.generated_path.read_text()
        required = (
            "source-vLLM sparse-retained paged KIVI4 flash decode",
            "extern \"C\" void jac_launch_sparse_retained(",
            "const int* RETAINED_BLOCK_TABLE",
            "int NUM_RETAINED_PAGES",
            "retained_block_table[logical_page]",
            "retained_old_valid",
            "residual_valid = bounded_seq_len - dense_old_valid",
            "<<<NH, 32, 0, stream>>>",
        )
        missing = [needle for needle in required if needle not in text]
        forbidden = ("int main(", "load_half(", "load_u8(", "int phys_page = block_table[logical_page]")
        present_forbidden = [needle for needle in forbidden if needle in text]
        if missing or present_forbidden:
            raise RuntimeError(
                "DSL sparse-retained KIVI decode artifact is not a vLLM-safe sparse library TU; "
                f"missing={missing} forbidden={present_forbidden}"
            )
        self.vllm_kernel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.generated_path, self.vllm_kernel_path)
        return self.vllm_kernel_path


CACHE_WRITE_SPEC = Path("specs/kernels/cache_write_kivi_paged.jac")
CACHE_WRITE_GENERATED_REL = Path(
    "generated/cache_write_kivi_paged/cache_write_kivi_paged.cu"
)
# NOT csrc/kivi/cache_write_kivi_dsl.cu: that path is already occupied by the EXISTING hand-CUDA
# writer (KIVI_CACHE_WRITE_SOURCE in source_patch.py, materialized there by an earlier
# session's scaffold naming -- "_dsl" in that name predates any actual DSL authorship and is
# misleading, but it is the real, served-path file and must not be silently overwritten by this
# artifact's own emit(). "_jac_dsl" here disambiguates: this is the jac-tool-generated version.
CACHE_WRITE_VLLM_KERNEL_REL = Path("csrc/kivi/cache_write_kivi_jac_dsl.cu")


@dataclass(frozen=True)
class DslCacheWriteKernelArtifact:
    """DSL-generated sibling of the hand-CUDA KIVI cache writer.

    Separate destination file (cache_write_kivi_jac_dsl.cu -- deliberately NOT
    cache_write_kivi_dsl.cu, the existing hand-CUDA writer's own materialized path; see
    CACHE_WRITE_VLLM_KERNEL_REL's own comment) because this is a different kernel entirely -- cache
    cache management (append + quantize-on-age-out + shift), not attention. Emitting this artifact does
    NOT make it the runtime's default source of truth: this writer carries no
    correctness claim in this repository, and source_patch can route
    single-token writes to it only under an explicit opt-in env. A GPU
    bitcheck plus a served parity/default-flip gate remain separate from
    materializing this artifact.
    """

    repo_root: str
    vllm_root: str
    spec: str = str(CACHE_WRITE_SPEC)
    variant: str = DSL_DECODE_VARIANT
    source_label: str = "dsl-kivi-flash-cache-write"
    generated_rel: str = str(CACHE_WRITE_GENERATED_REL)
    vllm_kernel_rel: str = str(CACHE_WRITE_VLLM_KERNEL_REL)

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def vllm_path(self) -> Path:
        return Path(self.vllm_root)

    @property
    def spec_path(self) -> Path:
        return self.repo_path / self.spec

    @property
    def generated_path(self) -> Path:
        return self.repo_path / self.generated_rel

    @property
    def vllm_kernel_path(self) -> Path:
        return self.vllm_path / self.vllm_kernel_rel

    def jac_emit_command(self) -> list[str]:
        return [
            "cargo", "run", "-q", "-p", "jac-cli", "--",
            "--emit-cuda", "--kernel-library", "--kivi-flash", self.spec,
        ]

    def report(self) -> dict[str, object]:
        return {
            **asdict(self),
            "spec_exists": self.spec_path.exists(),
            "generated_exists": self.generated_path.exists(),
            "vllm_kernel_exists": self.vllm_kernel_path.exists(),
            "jac_emit_command": self.jac_emit_command(),
            "serving_env": {
                "JAC_VLLM_KIVI_CACHE_WRITE_KERNEL_SOURCE": self.source_label,
                "JAC_VLLM_KIVI_CACHE_WRITE_KERNEL_CU": str(self.vllm_kernel_path),
            },
            "integrity_boundary": (
                "This only materializes the DSL-lowered CUDA source for the vLLM fork. "
                "This writer is not verified in this repository; the hand-CUDA "
                "kivi_cache_write_launch remains the served default, and source_patch "
                "exposes only an explicit opt-in selector for this one. It is not a "
                "served result or a default flip."
            ),
        }

    def emit(self) -> Path:
        if self.variant != DSL_DECODE_VARIANT:
            raise ValueError(f"only {DSL_DECODE_VARIANT!r} is supported, got {self.variant!r}")
        if not self.spec_path.exists():
            raise FileNotFoundError(f"missing KIVI cache-write DSL spec: {self.spec_path}")
        proc = subprocess.run(
            self.jac_emit_command(),
            cwd=self.repo_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if not self.generated_path.exists():
            raise FileNotFoundError(
                f"jacq CLI completed but did not write expected CUDA: {self.generated_path}"
            )
        text = self.generated_path.read_text()
        required = (
            "Library artifact for embedding in a runtime extension",
            "extern \"C\" void jac_launch_cache_write(",
            "source-vLLM paged KIVI4 cache writer",
            "static constexpr int NKV = 8;",
            "static constexpr int RESIDUAL_SCRATCH = 160;",
            "const int* BLOCK_TABLE",
            "if (next_old_tokens > prev_old_tokens)",
        )
        missing = [needle for needle in required if needle not in text]
        # Unlike the decode artifacts' forbidden lists, this omits "load_half(": the writer's own
        # {name}_load_half helper is a legitimate, intentional transcription of the hand-CUDA
        # source's own load_half helper (see kivi_cache_write_device_kernels), not a sign of an
        # accidentally-emitted wrong variant -- the writer has only one lowering (kivi_flash),
        # unlike decode's naive-vs-flash distinction that check exists to catch.
        forbidden = (
            "int main(", "load_u8(",
            "extern \"C\" void jac_launch(", "extern \"C\" void jac_launch_batched(",
        )
        present_forbidden = [needle for needle in forbidden if needle in text]
        if missing or present_forbidden:
            raise RuntimeError(
                "DSL KIVI cache-writer artifact is not a vLLM-safe library translation unit; "
                f"missing={missing} forbidden={present_forbidden}"
            )
        self.vllm_kernel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.generated_path, self.vllm_kernel_path)
        return self.vllm_kernel_path


CACHE_WRITE_BATCHED_SPEC = Path("specs/kernels/cache_write_kivi_paged_batched.jac")
CACHE_WRITE_BATCHED_GENERATED_REL = Path(
    "generated/cache_write_kivi_paged_batched/cache_write_kivi_paged_batched.cu"
)
CACHE_WRITE_BATCHED_VLLM_KERNEL_REL = Path("csrc/kivi/cache_write_kivi_batched_jac_dsl.cu")


@dataclass(frozen=True)
class DslCacheWriteBatchedKernelArtifact:
    """DSL-generated sibling of the hand-CUDA batched KIVI cache writer.

    This materializes a comparison translation unit for the batched prefill path. It is
    deliberately separate from both the served hand-CUDA writer (`cache_write_kivi_dsl.cu`) and the
    single-token DSL comparison artifact (`cache_write_kivi_jac_dsl.cu`). It is not wired into the
    request loop and carries no correctness claim until GPU-bitchecked against
    `kivi_cache_write_batched_dsl` over a multi-subchunk age-out schedule.
    """

    repo_root: str
    vllm_root: str
    spec: str = str(CACHE_WRITE_BATCHED_SPEC)
    variant: str = DSL_DECODE_VARIANT
    source_label: str = "dsl-kivi-flash-cache-write-batched"
    generated_rel: str = str(CACHE_WRITE_BATCHED_GENERATED_REL)
    vllm_kernel_rel: str = str(CACHE_WRITE_BATCHED_VLLM_KERNEL_REL)

    @property
    def repo_path(self) -> Path:
        return Path(self.repo_root)

    @property
    def vllm_path(self) -> Path:
        return Path(self.vllm_root)

    @property
    def spec_path(self) -> Path:
        return self.repo_path / self.spec

    @property
    def generated_path(self) -> Path:
        return self.repo_path / self.generated_rel

    @property
    def vllm_kernel_path(self) -> Path:
        return self.vllm_path / self.vllm_kernel_rel

    def jac_emit_command(self) -> list[str]:
        return [
            "cargo", "run", "-q", "-p", "jac-cli", "--",
            "--emit-cuda", "--kernel-library", "--kivi-flash", self.spec,
        ]

    def report(self) -> dict[str, object]:
        return {
            **asdict(self),
            "spec_exists": self.spec_path.exists(),
            "generated_exists": self.generated_path.exists(),
            "vllm_kernel_exists": self.vllm_kernel_path.exists(),
            "jac_emit_command": self.jac_emit_command(),
            "serving_env": {
                "JAC_VLLM_KIVI_CACHE_WRITE_BATCHED_KERNEL_SOURCE": self.source_label,
                "JAC_VLLM_KIVI_CACHE_WRITE_BATCHED_KERNEL_CU": str(self.vllm_kernel_path),
            },
            "integrity_boundary": (
                "This only materializes the DSL-lowered batched cache-writer CUDA source. It is "
                "not a CUDA bitcheck, not wired into the request loop, and not a served result. "
                "It must be GPU-bitchecked exactly against the hand-CUDA kivi_cache_write_batched_dsl "
                "op before it can replace that writer."
            ),
        }

    def emit(self) -> Path:
        if self.variant != DSL_DECODE_VARIANT:
            raise ValueError(f"only {DSL_DECODE_VARIANT!r} is supported, got {self.variant!r}")
        if not self.spec_path.exists():
            raise FileNotFoundError(f"missing batched KIVI cache-write DSL spec: {self.spec_path}")
        proc = subprocess.run(
            self.jac_emit_command(),
            cwd=self.repo_path,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        if not self.generated_path.exists():
            raise FileNotFoundError(
                f"jacq CLI completed but did not write expected CUDA: {self.generated_path}"
            )
        text = self.generated_path.read_text()
        required = (
            "Library artifact for embedding in a runtime extension",
            "extern \"C\" void jac_launch_cache_write_batched(",
            "source-vLLM BATCHED paged KIVI4 cache writer",
            "static constexpr int NKV = 8;",
            "static constexpr int MAX_NEW = 160;",
            "int START_SEQ_LEN, int NUM_NEW",
            "const int* BLOCK_TABLE",
            "append_scratch_tokens_batched_kernel",
            "if (next_old_tokens > prev_old_tokens)",
        )
        missing = [needle for needle in required if needle not in text]
        forbidden = (
            "int main(",
            "load_u8(",
            "extern \"C\" void jac_launch(",
            "extern \"C\" void jac_launch_batched(",
            "extern \"C\" void jac_launch_cache_write(\n",
        )
        present_forbidden = [needle for needle in forbidden if needle in text]
        if missing or present_forbidden:
            raise RuntimeError(
                "DSL batched KIVI cache-writer artifact is not a vLLM-safe library TU; "
                f"missing={missing} forbidden={present_forbidden}"
            )
        self.vllm_kernel_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.generated_path, self.vllm_kernel_path)
        return self.vllm_kernel_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--vllm-root", type=Path, required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--emit", action="store_true", help="run jacq codegen and copy CUDA into the vLLM source tree")
    ap.add_argument("--batched", action="store_true", help="emit the batched decode kernel (batched prefill) instead of the single-token one")
    ap.add_argument("--cache-write", action="store_true", help="emit the DSL cache-writer kernel instead of a decode kernel")
    ap.add_argument("--cache-write-batched", action="store_true", help="emit the DSL batched cache-writer kernel instead of a decode kernel")
    ap.add_argument("--sparse-retained", action="store_true", help="emit the sparse-retained decode kernel for page-aware KIVI eviction")
    ap.add_argument(
        "--tp2-local",
        action="store_true",
        help=(
            "also materialize the TP=2 local-rank decode variant (NH=20/NKV=4) "
            "with a distinct C symbol; valid for decode artifacts, not cache-write"
        ),
    )
    args = ap.parse_args()
    selected = [args.batched, args.cache_write, args.cache_write_batched, args.sparse_retained]
    if sum(bool(x) for x in selected) > 1:
        ap.error("--batched, --cache-write, --cache-write-batched, and --sparse-retained are mutually exclusive")
    if args.cache_write:
        if args.tp2_local:
            ap.error("--tp2-local is not valid with --cache-write; the served writer is runtime-NKV")
        cls = DslCacheWriteKernelArtifact
    elif args.cache_write_batched:
        if args.tp2_local:
            ap.error("--tp2-local is not valid with --cache-write-batched")
        cls = DslCacheWriteBatchedKernelArtifact
    elif args.sparse_retained:
        cls = DslDecodeSparseRetainedKernelArtifact
    elif args.batched:
        cls = DslDecodeBatchedKernelArtifact
    else:
        cls = DslDecodeKernelArtifact
    artifact = cls(
        repo_root=str(args.repo_root.resolve()),
        vllm_root=str(args.vllm_root.resolve()),
    )
    if args.emit:
        emitted = artifact.emit()
        if args.tp2_local:
            if args.sparse_retained:
                dst = artifact.vllm_path / TP2_LOCAL_SPARSE_RETAINED_VLLM_KERNEL_REL
                _materialize_tp2_local_variant(emitted, dst, symbol="sparse_retained")
            elif args.batched:
                dst = artifact.vllm_path / TP2_LOCAL_BATCHED_VLLM_KERNEL_REL
                _materialize_tp2_local_variant(emitted, dst, symbol="batched")
            else:
                dst = artifact.vllm_path / TP2_LOCAL_VLLM_KERNEL_REL
                _materialize_tp2_local_variant(emitted, dst, symbol="single")
    if args.json:
        report = artifact.report()
        if args.tp2_local:
            report["tp2_local_variant"] = {
                "nh": 20,
                "nkv": 4,
                "integrity_boundary": (
                    "TP2-local artifact materialization only; no CUDA bitcheck, "
                    "served HTTP proof, or capacity claim."
                ),
            }
            if args.sparse_retained:
                report["tp2_local_variant"]["vllm_kernel_rel"] = str(
                    TP2_LOCAL_SPARSE_RETAINED_VLLM_KERNEL_REL
                )
            elif args.batched:
                report["tp2_local_variant"]["vllm_kernel_rel"] = str(
                    TP2_LOCAL_BATCHED_VLLM_KERNEL_REL
                )
            else:
                report["tp2_local_variant"]["vllm_kernel_rel"] = str(TP2_LOCAL_VLLM_KERNEL_REL)
        print(json.dumps(report, indent=2, sort_keys=True))
    if not (args.json or args.emit):
        ap.error("choose --json and/or --emit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
