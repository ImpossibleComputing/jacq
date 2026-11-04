"""KIVI page layout math for a vLLM V1 source integration.

The point of this module is to make the KIVI/vLLM ABI explicit before touching
CUDA.  It mirrors the faithful KIVI storage layout:

* keys: int4 codes, grouped over tokens per channel;
* values: int4 codes, grouped over channels per token;
* scale+zero are fp16 pairs (4 bytes per group);
* the most recent residual window remains fp16.

For Qwen2.5-Coder-32B (8 KV heads, head_dim=128) with group=block_size=32, a
quantized page is 40,960 B/layer vs 131,072 B/layer fp16, i.e. 3.2x in the
old-page limit.  With residual=128 at 4096 context the effective ratio is the
~2.994x recorded ceiling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class KiviModelGeometry:
    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @classmethod
    def qwen25_coder_32b(cls) -> "KiviModelGeometry":
        return cls(
            name="Qwen2.5-Coder-32B-Instruct",
            num_layers=64,
            num_attention_heads=40,
            num_kv_heads=8,
            head_dim=128,
        )


@dataclass(frozen=True)
class KiviPageLayout:
    geometry: KiviModelGeometry
    block_size: int = 32
    group_size: int = 32
    nbits: int = 4
    residual_tokens: int = 128
    scale_zero_bytes: int = 4  # fp16 scale + fp16 zero

    def validate_for_vllm_lane(self) -> None:
        if self.nbits != 4:
            raise ValueError(f"first KIVI lane is int4 only, got nbits={self.nbits}")
        if self.block_size != self.group_size:
            raise ValueError(
                "KIVI key grouping must not straddle vLLM pages in the first "
                f"serving lane; require block_size==group_size, got "
                f"{self.block_size}!={self.group_size}"
            )
        if self.block_size != 32:
            raise ValueError(f"first vLLM KIVI lane requires block_size=32, got {self.block_size}")
        if self.geometry.head_dim % self.group_size != 0:
            raise ValueError(
                f"head_dim={self.geometry.head_dim} must be divisible by group_size={self.group_size}"
            )
        if self.residual_tokens % self.block_size != 0:
            raise ValueError(
                f"residual_tokens={self.residual_tokens} must be page-aligned for the first lane"
            )

    @property
    def elems_per_token(self) -> int:
        return self.geometry.num_kv_heads * self.geometry.head_dim

    @property
    def fp16_page_bytes_per_layer(self) -> int:
        # K + V, one fp16/bf16 element per cached component.
        return 2 * self.block_size * self.elems_per_token * self.geometry.dtype_bytes

    @property
    def code_page_bytes_per_component(self) -> int:
        return self.block_size * self.elems_per_token * self.nbits // 8

    @property
    def key_param_page_bytes(self) -> int:
        # One token group per block, one scale/zero pair per channel.
        token_groups = math.ceil(self.block_size / self.group_size)
        return token_groups * self.elems_per_token * self.scale_zero_bytes

    @property
    def value_param_page_bytes(self) -> int:
        # Per token, group along the channel/head_dim axis.
        channel_groups_per_token = self.geometry.num_kv_heads * math.ceil(
            self.geometry.head_dim / self.group_size
        )
        return self.block_size * channel_groups_per_token * self.scale_zero_bytes

    @property
    def quantized_page_bytes_per_layer(self) -> int:
        return (
            2 * self.code_page_bytes_per_component
            + self.key_param_page_bytes
            + self.value_param_page_bytes
        )

    @property
    def residual_pages(self) -> int:
        return self.residual_tokens // self.block_size

    def effective_bytes_per_layer(self, seq_len: int) -> int:
        pages = math.ceil(seq_len / self.block_size)
        residual_pages = min(self.residual_pages, pages)
        quantized_pages = max(0, pages - residual_pages)
        return (
            quantized_pages * self.quantized_page_bytes_per_layer
            + residual_pages * self.fp16_page_bytes_per_layer
        )

    def effective_bytes_per_token_all_layers(self, seq_len: int) -> float:
        return self.effective_bytes_per_layer(seq_len) * self.geometry.num_layers / seq_len

    def fp16_bytes_per_token_all_layers(self) -> int:
        return (
            2
            * self.elems_per_token
            * self.geometry.dtype_bytes
            * self.geometry.num_layers
        )

    def capacity_ratio_vs_fp16(self, seq_len: int) -> float:
        return self.fp16_bytes_per_token_all_layers() / self.effective_bytes_per_token_all_layers(seq_len)

    def to_abi_dict(self, seq_len: int = 4096) -> dict[str, object]:
        self.validate_for_vllm_lane()
        return {
            "geometry": asdict(self.geometry),
            "block_size": self.block_size,
            "group_size": self.group_size,
            "nbits": self.nbits,
            "residual_tokens": self.residual_tokens,
            "fp16_page_bytes_per_layer": self.fp16_page_bytes_per_layer,
            "quantized_page_bytes_per_layer": self.quantized_page_bytes_per_layer,
            "key_code_bytes": self.code_page_bytes_per_component,
            "value_code_bytes": self.code_page_bytes_per_component,
            "key_scale_zero_bytes": self.key_param_page_bytes,
            "value_scale_zero_bytes": self.value_param_page_bytes,
            "residual_pages": self.residual_pages,
            "effective_bytes_per_token_all_layers_at_seq_len": self.effective_bytes_per_token_all_layers(seq_len),
            "capacity_ratio_vs_fp16_at_seq_len": self.capacity_ratio_vs_fp16(seq_len),
            "vllm_allocator_requirement": (
                "not a homogeneous torch dtype page: needs K/V code tensors, K/V "
                "scale-zero tensors, and fp16 residual pages or side tensors"
            ),
        }
