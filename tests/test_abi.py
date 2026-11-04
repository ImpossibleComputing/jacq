import unittest

from integrations.vllm_kivi import abi
from integrations.vllm_kivi.layout import KiviModelGeometry


class AbiVersionTest(unittest.TestCase):
    def test_abi_version_is_explicit_and_positive(self):
        self.assertIsInstance(abi.JACQ_KIVI_ABI_VERSION, int)
        self.assertGreater(abi.JACQ_KIVI_ABI_VERSION, 0)

    def test_seq_len_cap_matches_recorded_abi_bound(self):
        # 124 old pages * 32 tokens + 128 residual = 4096 (the ABI v1 cap).
        self.assertEqual(abi.MAX_SEQ_LEN, 4096)
        self.assertEqual(abi.OLD_BLOCKS_CAP * abi.BLOCK_SIZE + abi.RESIDUAL_TOKENS, 4096)

    def test_required_torch_ops_cover_decode_and_both_writers(self):
        self.assertEqual(len(abi.REQUIRED_TORCH_OPS), 3)
        self.assertIn("kivi_decode_dsl", abi.REQUIRED_TORCH_OPS)
        self.assertIn("kivi_cache_write_dsl", abi.REQUIRED_TORCH_OPS)
        self.assertIn("kivi_cache_write_jac_dsl", abi.REQUIRED_TORCH_OPS)

    def test_cache_write_selector_is_fail_closed(self):
        self.assertEqual(abi.CACHE_WRITE_KERNELS, ("hand_cuda", "jac_dsl"))


class AbiFailClosedTest(unittest.TestCase):
    def test_seq_len_above_cap_fails_closed(self):
        with self.assertRaises(abi.UnsupportedAbiError):
            abi.kernel_interface(seq_len=4097)

    def test_seq_len_at_cap_is_reported(self):
        report = abi.kernel_interface(seq_len=4096)
        self.assertEqual(report["abi_version"], abi.JACQ_KIVI_ABI_VERSION)
        self.assertEqual(report["seq_len"], 4096)
        self.assertEqual(report["old_pages"], 124)

    def test_vllm_version_mismatch_fails_closed(self):
        for bad in ("0.9.1", "0.8.4", "0.9.5"):
            with self.assertRaises(abi.UnsupportedAbiError, msg=bad):
                abi.check_vllm_version(bad)
        # the supported line and an exact source-head match must pass
        abi.check_vllm_version("0.8.5.post2.dev0+g3015d5634.d20260815", abi.SUPPORTED_VLLM_SOURCE_HEAD)
        # bare supported version without a head is allowed (the head check
        # is a separate, explicit preflight; see docs/VLLM_SETUP.md)
        abi.check_vllm_version("0.8.5")

    def test_vllm_version_lookalike_prefixes_fail_closed(self):
        # regression: a plain startswith("0.8.5") check accepted these
        for lookalike in ("0.8.50", "0.8.55", "0.8.5.1", "0.8.5x", "0.8.5-rc1+gdeadbeef"):
            with self.assertRaises(abi.UnsupportedAbiError, msg=lookalike):
                abi.check_vllm_version(lookalike)

    def test_supported_version_suffixes_pass(self):
        for good in (
            "0.8.5",
            "0.8.5.post2",
            "0.8.5.post2.dev0",
            "0.8.5.post2.dev0+g3015d5634.d20260815",
        ):
            abi.check_vllm_version(good)

    def test_wrong_source_head_fails_closed(self):
        with self.assertRaises(abi.UnsupportedAbiError):
            abi.check_vllm_version("0.8.5", "0" * 40)

    def test_unvalidated_cuda_arch_fails_closed(self):
        with self.assertRaises(abi.UnsupportedAbiError):
            abi.check_cuda_arch("sm_75")
        abi.check_cuda_arch("sm_80")
        abi.check_cuda_arch("sm_89")

    def test_foreign_geometry_fails_closed(self):
        foreign = KiviModelGeometry(
            name="other", num_layers=32, num_attention_heads=32, num_kv_heads=8, head_dim=128
        )
        with self.assertRaises(abi.UnsupportedAbiError):
            abi.check_geometry(foreign)
        abi.check_geometry(abi.supported_geometry())

    def test_decode_launch_order_matches_bindings(self):
        report = abi.kernel_interface()
        decode = [k for k in report["kernels"] if k["torch_op"] == "kivi_decode_dsl"][0]
        self.assertEqual(decode["launch_order"][-1], "cudaStream_t stream")
        self.assertIn("SEQ_LEN", decode["launch_order"])
        self.assertIn("BLOCK_TABLE", decode["launch_order"])

    def test_report_states_integrity_boundary(self):
        self.assertIn("NOT RUN", abi.kernel_interface()["integrity_boundary"])


if __name__ == "__main__":
    unittest.main()
