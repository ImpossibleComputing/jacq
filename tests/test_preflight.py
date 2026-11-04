"""Integration tests for the fail-closed preflight (blocker: checks must be
wired to a runnable path, not just be standalone helpers).

The synthetic-tree helper is shared with test_layout_and_scaffold. The git
HEAD accessor is mocked ONLY where a synthetic tree cannot carry the pinned
HEAD; the real pinned-head behavior is demonstrated by the real-tree test
below and by `--preflight` on an actual checkout (no GPU required).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from integrations.vllm_kivi import abi, preflight
from test_layout_and_scaffold import _write_minimal_vllm_tree

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED = abi.SUPPORTED_VLLM_SOURCE_HEAD


def make_tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="jacq_preflight_"))
    _write_minimal_vllm_tree(root)
    return root


class PreflightUnitTest(unittest.TestCase):
    def test_all_pass_when_pinned_head_anchors_and_arch_ok(self):
        tree = make_tree()
        with mock.patch.object(preflight, "git_head", return_value=PINNED), \
             mock.patch.object(preflight, "detect_cuda_arch", return_value="sm_89"):
            report = preflight.preflight(tree)
        self.assertTrue(report["ok"])
        statuses = {c["name"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses["git_head_pinned"], "pass")
        self.assertEqual(statuses["scaffold_anchors"], "pass")
        self.assertEqual(statuses["cuda_arch"], "pass")
        # the runtime-only boundary is reported honestly, not claimed as checked
        self.assertEqual(statuses["model_geometry_and_shapes"], "deferred-to-runtime")

    def test_drifted_head_fails_closed(self):
        tree = make_tree()
        with mock.patch.object(preflight, "git_head", return_value="f" * 40), \
             mock.patch.object(preflight, "detect_cuda_arch", return_value="sm_89"):
            report = preflight.preflight(tree)
        self.assertFalse(report["ok"])
        check = [c for c in report["checks"] if c["name"] == "git_head_pinned"][0]
        self.assertIn("not the pinned head", check["detail"])

    def test_tarball_without_git_fails_closed(self):
        tree = make_tree()
        with mock.patch.object(preflight, "git_head", return_value=None), \
             mock.patch.object(preflight, "detect_cuda_arch", return_value="sm_80"):
            report = preflight.preflight(tree)
        self.assertFalse(report["ok"])
        self.assertIn("not a git repository", report["checks"][0]["detail"])

    def test_unvalidated_cuda_arch_fails_closed(self):
        tree = make_tree()
        with mock.patch.object(preflight, "git_head", return_value=PINNED):
            report = preflight.preflight(tree, cuda_arch="sm_75")
        self.assertFalse(report["ok"])
        check = [c for c in report["checks"] if c["name"] == "cuda_arch"][0]
        self.assertEqual(check["status"], "fail")

    def test_missing_cuda_arch_fails_closed(self):
        tree = make_tree()
        with mock.patch.object(preflight, "git_head", return_value=PINNED), \
             mock.patch.object(preflight, "detect_cuda_arch", return_value=None):
            report = preflight.preflight(tree)
        self.assertFalse(report["ok"])
        self.assertIn("--cuda-arch", report["checks"][2]["detail"])

    def test_anchor_drift_fails_without_mock(self):
        tree = make_tree()
        # corrupt one anchored file so inspect_checkout must fail, no mocks
        (tree / "vllm/config.py").write_text("drifted")
        with mock.patch.object(preflight, "git_head", return_value=PINNED), \
             mock.patch.object(preflight, "detect_cuda_arch", return_value="sm_89"):
            report = preflight.preflight(tree)
        self.assertFalse(report["ok"])
        check = [c for c in report["checks"] if c["name"] == "scaffold_anchors"][0]
        self.assertEqual(check["status"], "fail")


class PreflightCLITest(unittest.TestCase):
    def run_cli(self, *extra):
        return subprocess.run(
            [sys.executable, "-m", "integrations.vllm_kivi.source_patch"] + list(extra),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    def test_apply_refuses_to_run_without_preflight_pass(self):
        tree = make_tree()
        r = self.run_cli("--vllm-root", str(tree), "--apply")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("preflight failed", r.stdout + r.stderr)

    def test_apply_still_works_with_explicit_test_harness_escape(self):
        tree = make_tree()
        r = self.run_cli("--vllm-root", str(tree), "--apply", "--skip-preflight")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((tree / "vllm/v1/attention/backends/kivi_dsl.py").exists())

    def test_preflight_flag_reports_json_and_exit_code(self):
        tree = make_tree()
        r = self.run_cli("--vllm-root", str(tree), "--preflight", "--cuda-arch", "sm_89")
        self.assertNotEqual(r.returncode, 0)
        report = json.loads(r.stdout)
        self.assertFalse(report["ok"])
        # synthetic tree has no git repo and no nvidia-smi in this container
        names = {c["name"]: c["status"] for c in report["checks"]}
        self.assertEqual(names["git_head_pinned"], "fail")
        self.assertEqual(names["cuda_arch"], "pass")  # explicit --cuda-arch sm_89

    def test_preflight_on_real_pinned_tarball_fails_closed(self):
        """Real-tree evidence (CPU-only): the actual pinned-head source tree
        WITHOUT .git must fail preflight — a tarball cannot prove its commit."""
        real = Path("/opt/data/cache/scratch/vllm-src/tree")
        if not real.exists():
            self.skipTest("pinned vLLM tarball not present")
        r = self.run_cli("--vllm-root", str(real), "--preflight", "--cuda-arch", "sm_89")
        self.assertNotEqual(r.returncode, 0)
        report = json.loads(r.stdout)
        names = {c["name"]: c["status"] for c in report["checks"]}
        self.assertEqual(names["git_head_pinned"], "fail")  # no .git: correct
        self.assertEqual(names["scaffold_anchors"], "pass")  # anchors real-pass


if __name__ == "__main__":
    unittest.main()
