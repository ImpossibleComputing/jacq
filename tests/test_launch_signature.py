"""Pin the Python binding order to the ACTUAL emitted CUDA launch signature.

This test parses `extern "C" void jac_launch(...)` from the checked-in
codegen snapshot (an emitted artifact, regenerated only via the compiler) and
compares its parameter order to `bindings.jac_launch_order()`. It guards
against helper/ABI drift between the DSL codegen and the Python binding
layer. It does NOT execute CUDA and makes no runtime-correctness claim.
"""
import re
import unittest
from pathlib import Path

from integrations.vllm_kivi.bindings import default_qwen32b_binding_abi

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = (
    REPO_ROOT
    / "crates/jac-codegen-cuda/tests/snapshots/kernels/decode_attention_kivi_paged_library_flash.cu"
)
SPEC = REPO_ROOT / "specs/kernels/decode_attention_kivi_paged.jac"


def snapshot_launch_params() -> list[str]:
    cu = SNAPSHOT.read_text()
    m = re.search(r'extern "C" void jac_launch\((.*?)\)\s*\{', cu, re.DOTALL)
    assert m, "jac_launch signature not found in snapshot"
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    names = []
    for p in params:
        # strip C/C++ type constructs, keep the parameter identifier
        ident = re.findall(r"[A-Za-z_][A-Za-z0-9_]*$", p.replace("const ", "").replace("*", "").strip())
        names.append(ident[0])
    return names


class LaunchSignatureTest(unittest.TestCase):
    def test_binding_order_matches_emitted_cuda_signature(self):
        emitted = snapshot_launch_params()
        self.assertEqual(
            emitted,
            [
                "Q", "BLOCK_TABLE",
                "K_CODE_PAGES", "K_SCALE_PAGES", "K_ZERO_PAGES",
                "V_CODE_PAGES", "V_SCALE_PAGES", "V_ZERO_PAGES",
                "K_RESIDUAL_SCRATCH", "V_RESIDUAL_SCRATCH",
                "SEQ_LEN", "O", "stream",
            ],
            "emitted jac_launch parameter order changed; update bindings.py and ABI version",
        )
        binding = default_qwen32b_binding_abi()
        expected = binding.jac_launch_order()
        # binding list carries the stream entry in full C form
        self.assertEqual(expected[-1], "cudaStream_t stream")
        self.assertEqual(expected[:-1], emitted[:-1], "bindings.jac_launch_order disagrees with emitted CUDA")
        # regression: the stale non-scratch residual tensors must stay gone
        self.assertNotIn("K_RESIDUAL", expected)
        self.assertNotIn("V_RESIDUAL", expected)

    def test_abi_kernel_interface_agrees_with_emitted_signature(self):
        from integrations.vllm_kivi import abi

        report = abi.kernel_interface()
        decode = [k for k in report["kernels"] if k["torch_op"] == "kivi_decode_dsl"][0]
        emitted = snapshot_launch_params()
        got = decode["launch_order"]
        self.assertEqual(got[-1], "cudaStream_t stream")
        self.assertEqual(got[:-1], emitted[:-1])

    def test_spec_has_no_plain_residual_tensors(self):
        spec = SPEC.read_text()
        # the tensors block must not declare K_RESIDUAL/V_RESIDUAL (only *_SCRATCH)
        self.assertNotIn(
            "K_RESIDUAL[", spec.replace("K_RESIDUAL_SCRATCH[", "").replace("V_RESIDUAL_SCRATCH[", "")
        )
        self.assertNotIn(
            "V_RESIDUAL[", spec.replace("K_RESIDUAL_SCRATCH[", "").replace("V_RESIDUAL_SCRATCH[", "")
        )


if __name__ == "__main__":
    unittest.main()
