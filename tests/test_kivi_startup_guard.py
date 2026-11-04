"""Executable checks of the emitted vLLM int4-KIVI startup gate."""
import types
import unittest

from integrations.vllm_kivi import source_patch


class KiviStartupGuardTest(unittest.TestCase):
    def setUp(self):
        self.envs = types.SimpleNamespace(
            VLLM_USE_V1=True, VLLM_ATTENTION_BACKEND=None)
        namespace = {
            "envs": self.envs,
            "torch": types.SimpleNamespace(float16="float16"),
        }
        exec(source_patch.KIVI_STARTUP_GUARD_SOURCE, namespace)
        self.validate = namespace["_validate_int4_kivi_compatibility"]
        self.model = types.SimpleNamespace(
            architectures=["Qwen2ForCausalLM"],
            hf_text_config=types.SimpleNamespace(
                model_type="qwen2", num_attention_heads=40),
            use_mla=False, dtype="float16", max_model_len=512,
            get_hf_config_sliding_window=lambda: None,
            get_total_num_kv_heads=lambda: 8,
            get_head_size=lambda: 128,
        )
        self.cache = types.SimpleNamespace(
            cache_dtype="int4_kivi", block_size=32,
            enable_prefix_caching=False)
        self.parallel = types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1)

    def test_supported_qwen_global_geometry_and_tp2(self):
        self.validate(self.model, self.cache, self.parallel)
        self.model.max_model_len = 4096
        self.parallel.tensor_parallel_size = 2
        self.validate(self.model, self.cache, self.parallel)

    def test_stock_cache_is_unchanged(self):
        self.cache.cache_dtype = "auto"
        self.validate(None, self.cache, None)
        self.validate(None, None, None)

    def test_mistral_and_qwen3_rejected_before_runtime(self):
        for architecture, model_type in (
            ("MistralForCausalLM", "mistral"),
            ("Qwen3ForCausalLM", "qwen3"),
        ):
            with self.subTest(architecture=architecture):
                self.model.architectures = [architecture]
                self.model.hf_text_config.model_type = model_type
                with self.assertRaisesRegex(ValueError, "only Qwen2ForCausalLM"):
                    self.validate(self.model, self.cache, self.parallel)

    def test_qwen_wrong_geometry_rejected(self):
        self.model.hf_text_config.num_attention_heads = 14
        self.model.get_total_num_kv_heads = lambda: 2
        self.model.get_head_size = lambda: 64
        with self.assertRaisesRegex(ValueError, "got 14/2/64"):
            self.validate(self.model, self.cache, self.parallel)

    def test_unsupported_backend_and_features_rejected(self):
        checks = (
            (lambda: setattr(self.envs, "VLLM_USE_V1", False), "requires the Jacq-patched vLLM V1"),
            (lambda: setattr(self.envs, "VLLM_ATTENTION_BACKEND", "FLASHINFER"), "VLLM_ATTENTION_BACKEND"),
            (lambda: setattr(self.model, "dtype", "bfloat16"), "fp16 activations"),
            (lambda: setattr(self.model, "max_model_len", 4097), "max-model-len"),
            (lambda: setattr(self.cache, "block_size", 16), "block-size 32"),
            (lambda: setattr(self.cache, "enable_prefix_caching", True), "no-enable-prefix-caching"),
            (lambda: setattr(self.model, "use_mla", True), "MLA or sliding-window"),
            (lambda: setattr(self.model, "get_hf_config_sliding_window", lambda: 4096), "MLA or sliding-window"),
            (lambda: setattr(self.parallel, "tensor_parallel_size", 3), "TP=1 or TP=2"),
            (lambda: setattr(self.parallel, "pipeline_parallel_size", 2), "PP=1"),
        )
        for mutate, message in checks:
            with self.subTest(message=message):
                self.setUp()
                mutate()
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(self.model, self.cache, self.parallel)

if __name__ == "__main__":
    unittest.main()
