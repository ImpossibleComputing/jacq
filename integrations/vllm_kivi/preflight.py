"""Fail-closed preflight for the KIVI/vLLM serving lane.

Runs the real compatibility checks from `abi.py` against an actual vLLM
checkout BEFORE the scaffold is applied or built:

1. the checkout is a git repository whose HEAD is exactly the pinned source
   head (a source tarball or drifted clone fails closed — the version string
   in a checkout is build-generated, so the commit is the only verifiable
   identity at preflight time);
2. the scaffold anchors match (reuses `source_patch.inspect_checkout`);
3. the CUDA architecture is one of the validated set (via `--cuda-arch`, or
   auto-detected from `nvidia-smi` when it reports exactly one compute
   capability). This is a declared/detected BUILD-TARGET check, not proof
   of the runtime GPU: it validates what the caller declares or what a
   single-architecture host reports, and heterogeneous multi-GPU hosts are
   not silently resolved to one arch. The ACTUAL serving device is
   enforced fail-closed at the C++ torch-op launch boundary
   (`check_launch_device` in `KIVI_BINDING_SOURCE`,
   `integrations/vllm_kivi/source_patch.py`), which reads the compute
   capability of the device the tensors are bound to, accepts only
   sm_80/sm_89, and rejects mixed-device launches.

Model geometry and per-request shapes are validated at runtime by the
scaffolded attention backend when tensors are bound; the preflight reports
that boundary as `deferred-to-runtime` rather than pretending to check it.

`source_patch.py --preflight` exercises this. `--apply` runs the same
preflight first and refuses to proceed on failure (``--skip-preflight``
exists only for CPU tests against synthetic trees and must not be used on a
real checkout).

This is a preflight, not evidence: passing it proves the checkout/anchors/
arch are compatible, not that kernels are correct or serving was exercised.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from integrations.vllm_kivi import abi
from integrations.vllm_kivi.source_patch import inspect_checkout


def git_head(root: Path) -> str | None:
    """Resolve `git rev-parse HEAD` under `root`; None when not a git repo."""
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def detect_cuda_arch() -> str | None:
    """Detect the local GPU compute capability via nvidia-smi, if available.

    Returns e.g. 'sm_89', or None when nvidia-smi is absent/unhelpful (e.g.
    in a CPU-only container) or when it reports heterogeneous compute
    capabilities. Heterogeneous GPUs are deliberately NOT resolved to one
    arch: silently picking one would launder a multi-GPU host through a
    single-arch build-target check, so the caller must pass --cuda-arch
    explicitly instead.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    proc = subprocess.run(
        [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    caps = {c.strip().replace(".", "") for c in proc.stdout.splitlines() if c.strip()}
    if len(caps) != 1:
        return None
    return "sm_" + next(iter(caps))


def preflight(vllm_root: Path, cuda_arch: str | None = None) -> dict[str, object]:
    """Run all preflight checks; report `ok=False` (and raise nothing) when
    any hard boundary fails, so callers can print the full report."""
    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str) -> bool:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})
        return ok

    # 1. pinned HEAD
    head = git_head(vllm_root)
    head_ok = head is not None and head == abi.SUPPORTED_VLLM_SOURCE_HEAD
    if head is None:
        head_detail = (
            "not a git repository (or git failed): cannot verify the pinned "
            f"source head {abi.SUPPORTED_VLLM_SOURCE_HEAD[:12]}...; clone the "
            "pinned vLLM revision instead of using a source tarball"
        )
    elif head_ok:
        head_detail = f"HEAD matches the pinned head; vLLM {abi.SUPPORTED_VLLM_VERSION} line implied"
    else:
        head_detail = (
            f"HEAD {head[:12]}... is not the pinned head "
            f"{abi.SUPPORTED_VLLM_SOURCE_HEAD[:12]}...; run "
            f"`git -C {vllm_root} checkout {abi.SUPPORTED_VLLM_SOURCE_HEAD}`"
        )
    record("git_head_pinned", head_ok, head_detail)
    if head_ok:
        # explicit re-run of the version check for machine consumers
        abi.check_vllm_version(abi.SUPPORTED_VLLM_VERSION, source_head=head)

    # 2. anchors
    inspect = inspect_checkout(vllm_root)
    anchors_ok = bool(inspect.get("ok_to_apply_scaffold"))
    record(
        "scaffold_anchors",
        anchors_ok,
        "all scaffold anchors matched" if anchors_ok else "anchor drift: run --json for the full anchor report",
    )

    # 3. CUDA architecture (declared/detected BUILD TARGET only)
    arch = cuda_arch or detect_cuda_arch()
    if arch is None:
        arch_ok = False
        arch_detail = (
            "CUDA build target not verified: no single-architecture "
            "nvidia-smi report and no --cuda-arch; heterogeneous GPU hosts "
            "are not silently resolved to one arch -- pass --cuda-arch "
            f"explicitly (validated targets: {', '.join(abi.SUPPORTED_CUDA_ARCHES)})"
        )
    else:
        try:
            abi.check_cuda_arch(arch)
            arch_ok = True
            arch_detail = (
                f"build target {arch} is in the validated set "
                "(declared/detected target only -- it does not prove the "
                "runtime GPU; the actual serving device is checked "
                "fail-closed at the C++ torch-op launch boundary)"
            )
        except abi.UnsupportedAbiError as e:
            arch_ok, arch_detail = False, str(e)
    record("cuda_arch", arch_ok, arch_detail)

    checks.append(
        {
            "name": "model_geometry_and_shapes",
            "status": "deferred-to-runtime",
            "detail": (
                "geometry/seq_len/block-size are validated by the scaffolded "
                "backend when tensors are bound, not by this preflight"
            ),
        }
    )

    return {
        "ok": head_ok and anchors_ok and arch_ok,
        "abi_version": abi.JACQ_KIVI_ABI_VERSION,
        "vllm_root": str(vllm_root),
        "checks": checks,
        "integrity_boundary": (
            "Preflight verifies checkout identity, scaffold anchors, and a "
            "declared/detected CUDA build target only. It does not build, "
            "launch CUDA, or establish kernel correctness or any serving "
            "result. The actual serving GPU is enforced fail-closed at the "
            "C++ torch-op launch boundary (check_launch_device), not here."
        ),
    }


def run_preflight_cli(vllm_root: Path, cuda_arch: str | None) -> int:
    report = preflight(vllm_root, cuda_arch)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1
