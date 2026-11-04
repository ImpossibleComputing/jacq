"""Versioned callable CUDA kernel ABI for the jacq KIVI serving lane.

This module is the single compatibility surface between three independently
built artifacts:

1. the DSL-generated CUDA translation units emitted by `jacq --emit-cuda
   --kernel-library` from `specs/kernels/*.jac`;
2. the vLLM source scaffold applied by `integrations.vllm_kivi.source_patch`
   to a pinned vLLM checkout;
3. the Python/torch call sites in the scaffolded vLLM tree.

Everything here fails closed: an unsupported vLLM version, CUDA architecture,
model geometry, or sequence length raises `UnsupportedAbiError` with the exact
boundary that was violated instead of silently degrading. The ABI version is
bumped when any of the constants, tensor shapes, or the launch order change;
consumers must compare `JACQ_KIVI_ABI_VERSION` before binding tensors.

Bounded by design target: the constants below describe the lane this ABI
was built for (Qwen2.5-Coder attention geometry, vLLM 0.8.5, A100/RTX 4090).
They are limits, not performance claims; no GPU or serving evidence is
reproducible from this repository (see docs/LIMITATIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from integrations.vllm_kivi.bindings import KiviDecodeBindingAbi, default_qwen32b_binding_abi
from integrations.vllm_kivi.layout import KiviModelGeometry, KiviPageLayout


class UnsupportedAbiError(Exception):
    """Raised when a request or environment violates a hard ABI boundary."""


# Bumped on any change to tensor shapes, launch order, or the constants below.
JACQ_KIVI_ABI_VERSION = 1

# The vLLM source tree the scaffold patch anchors against (v0.8.5 line).
SUPPORTED_VLLM_VERSION = "0.8.5"
SUPPORTED_VLLM_SOURCE_HEAD = "3015d5634e74d59704e2b39bab0dbe2e6f86a38a"

# Validated compute capabilities: sm_80 (A100), sm_89 (RTX 4090). This tuple
# is the declared build-target set checked by preflight; the ACTUAL serving
# device is enforced fail-closed at the C++ torch-op launch boundary
# (check_launch_device in source_patch.py's KIVI_BINDING_SOURCE).
SUPPORTED_CUDA_ARCHES = ("sm_80", "sm_89")

# ABI cap recorded in the mechanism preflight: 124 old pages of 32 tokens plus
# a 128-token fp16 residual window => 4096 sequence tokens maximum.
OLD_BLOCKS_CAP = 124
BLOCK_SIZE = 32
RESIDUAL_TOKENS = 128
MAX_SEQ_LEN = OLD_BLOCKS_CAP * BLOCK_SIZE + RESIDUAL_TOKENS  # 4096

# torch ops the scaffolded vLLM build must expose via vllm._C before any
# jacq kernel is callable.
REQUIRED_TORCH_OPS = (
    "kivi_cache_write_dsl",
    "kivi_cache_write_jac_dsl",
    "kivi_decode_dsl",
)

# Cache-writer selector read by the scaffolded backend; anything else fails.
CACHE_WRITE_KERNELS = ("hand_cuda", "jac_dsl")


def supported_geometry() -> KiviModelGeometry:
    """The only validated attention geometry for the serving lane."""
    return KiviModelGeometry.qwen25_coder_32b()


@dataclass(frozen=True)
class KernelCall:
    """One callable entry point of the ABI, with its launch order."""

    torch_op: str
    spec: str
    launch_order: tuple[str, ...] = ()

    def describe(self) -> dict[str, object]:
        return {
            "torch_op": self.torch_op,
            "spec": self.spec,
            "launch_order": list(self.launch_order),
        }


def kernel_interface(seq_len: int = MAX_SEQ_LEN) -> dict[str, object]:
    """The versioned, callable kernel surface for one decode step.

    Returns a JSON-serialisable description of every op, its generating spec
    file, and the exact tensor launch order a caller must bind. Raises
    `UnsupportedAbiError` if `seq_len` exceeds the ABI cap (fail closed).
    """
    abi = _checked_binding(seq_len)
    return {
        "abi_version": JACQ_KIVI_ABI_VERSION,
        "supported_vllm_version": SUPPORTED_VLLM_VERSION,
        "supported_cuda_arches": list(SUPPORTED_CUDA_ARCHES),
        "required_torch_ops": list(REQUIRED_TORCH_OPS),
        "cache_write_kernels": list(CACHE_WRITE_KERNELS),
        "seq_len": abi.seq_len,
        "old_pages": abi.old_pages,
        "kernels": [
            KernelCall(
                torch_op="kivi_decode_dsl",
                spec="specs/kernels/decode_attention_kivi_paged.jac",
                launch_order=tuple(abi.jac_launch_order()),
            ).describe(),
            KernelCall(
                torch_op="kivi_cache_write_jac_dsl",
                spec="specs/kernels/cache_write_kivi_paged.jac",
                launch_order=(),
            ).describe(),
        ],
        "integrity_boundary": (
            "This describes the callable surface only. It does not allocate "
            "tensors, launch CUDA, or make any throughput/quality claim. "
            "GPU execution was NOT RUN in the repository build environment; "
            "see docs/LIMITATIONS.md."
        ),
    }


def _checked_binding(seq_len: int) -> KiviDecodeBindingAbi:
    if seq_len > MAX_SEQ_LEN:
        raise UnsupportedAbiError(
            f"seq_len {seq_len} exceeds the KIVI ABI cap of {MAX_SEQ_LEN} "
            f"({OLD_BLOCKS_CAP} old pages x {BLOCK_SIZE} tokens + "
            f"{RESIDUAL_TOKENS}-token residual). Long-context support "
            "requires an explicit ABI lift."
        )
    return default_qwen32b_binding_abi(seq_len=seq_len)


def check_vllm_version(version: str, source_head: str | None = None) -> None:
    """Fail closed unless the vLLM build matches the supported source line.

    Accepts exactly the `0.8.5` line: `0.8.5` itself plus PEP 440-style
    non-numeric suffix segments (`0.8.5.post2.dev0+g3015d5634.d20260815`).
    Lookalike prefixes such as `0.8.50` or `0.8.55` are rejected — a plain
    `startswith` check would accept them, contradicting fail-closed.

    `source_head` is optional because a version string does not prove which
    commit is checked out. The caller SHOULD resolve the actual Git HEAD
    (`git -C <vllm-root> rev-parse HEAD`) and pass it; passing a mismatched
    or absent head is then an explicit, reportable condition rather than a
    silently skipped check.
    """
    if not _on_supported_version_line(version):
        raise UnsupportedAbiError(
            f"vLLM {version} is not supported; this ABI targets exactly vLLM "
            f"{SUPPORTED_VLLM_VERSION} (source head "
            f"{SUPPORTED_VLLM_SOURCE_HEAD[:12]}...). Refusing to continue."
        )
    if source_head is not None and source_head != SUPPORTED_VLLM_SOURCE_HEAD:
        raise UnsupportedAbiError(
            f"vLLM source head {source_head[:12]}... does not match the "
            f"pinned scaffold head {SUPPORTED_VLLM_SOURCE_HEAD[:12]}...; "
            "resolve the checkout with `git -C <vllm-root> rev-parse HEAD` "
            "and check it against the pinned head before building."
        )


def _on_supported_version_line(version: str) -> bool:
    core = version.split("+", 1)[0].split(".")
    if len(core) < 3 or core[:3] != ["0", "8", "5"]:
        return False
    # After the patch number only non-numeric suffix segments are allowed
    # (post2, dev0, rc1, ...). Numeric segments (0.8.5.1) are a different
    # release line and fail closed.
    return len(core) == 3 or all(not seg[:1].isdigit() for seg in core[3:])


def check_cuda_arch(arch: str) -> None:
    """Fail closed on declared/detected CUDA build targets outside the set.

    `arch` is a build target (a caller-declared `--cuda-arch` or a value
    auto-detected from `nvidia-smi`), not proof of which GPU a request will
    actually serve. The actual tensor device is enforced at the C++ torch-op
    launch boundary (see `check_launch_device` in
    `integrations.vllm_kivi/source_patch.py`'s `KIVI_BINDING_SOURCE`).
    """
    if arch not in SUPPORTED_CUDA_ARCHES:
        raise UnsupportedAbiError(
            f"CUDA arch {arch} is outside the validated set "
            f"{SUPPORTED_CUDA_ARCHES} (A100, RTX 4090). Refusing to run."
        )


def check_geometry(geometry: KiviModelGeometry) -> None:
    """Fail closed on attention geometries the kernels have not been bound for."""
    expected = supported_geometry()
    for attr in ("num_attention_heads", "num_kv_heads", "head_dim"):
        got = getattr(geometry, attr)
        want = getattr(expected, attr)
        if got != want:
            raise UnsupportedAbiError(
                f"model geometry {attr}={got} does not match the supported "
                f"lane ({expected.name}: {attr}={want}); fail closed."
            )
