"""Tensor binding ABI for the source-vLLM int4-KIVI decode lane.

This module is intentionally framework-light: it does not allocate tensors or call CUDA.  It is the
single source of truth for the shapes/dtypes/order that the vLLM source fork must bind before it can
call the DSL-lowered `jac_launch` from `csrc/kivi/decode_attention_kivi_dsl.cu`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from integrations.vllm_kivi.layout import KiviModelGeometry, KiviPageLayout


@dataclass(frozen=True)
class TensorBinding:
    name: str
    dtype: str
    shape: tuple[int | str, ...]
    bytes_per_element: int
    role: str

    @property
    def numel_per_layer(self) -> int | str:
        prod = 1
        for dim in self.shape:
            if isinstance(dim, str):
                return "dynamic"
            prod *= dim
        return prod

    @property
    def bytes_per_layer(self) -> int | str:
        n = self.numel_per_layer
        if isinstance(n, str):
            return n
        return n * self.bytes_per_element

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "shape": list(self.shape),
            "numel_per_layer": self.numel_per_layer,
            "bytes_per_layer": self.bytes_per_layer,
        }


@dataclass(frozen=True)
class KiviDecodeBindingAbi:
    """Paged KIVI tensors consumed by the DSL-lowered `jac_launch`.

    `num_blocks` is the physical page-pool size for the one-layer binding report.  For the first
    4096-token smoke this defaults to the quantized old-token pages only: `(4096 - residual) / 32 =
    124`.  A production vLLM pool may allocate more pages; the shape contract is unchanged.
    """

    layout: KiviPageLayout
    seq_len: int = 4096
    num_blocks: int | None = None
    max_blocks_per_seq: int | None = None

    def __post_init__(self) -> None:
        self.layout.validate_for_vllm_lane()
        if self.seq_len < self.layout.residual_tokens:
            raise ValueError("seq_len must cover the fp16 residual window in the first KIVI lane")
        if self.seq_len % self.layout.block_size != 0:
            raise ValueError("first KIVI binding ABI expects page-aligned seq_len")
        if self.num_blocks is not None and self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.max_blocks_per_seq is not None and self.max_blocks_per_seq <= 0:
            raise ValueError("max_blocks_per_seq must be positive")

    @property
    def old_tokens(self) -> int:
        return max(0, self.seq_len - self.layout.residual_tokens)

    @property
    def old_pages(self) -> int:
        return self.old_tokens // self.layout.block_size

    @property
    def physical_blocks(self) -> int:
        return self.num_blocks if self.num_blocks is not None else self.old_pages

    @property
    def block_table_len(self) -> int:
        return self.max_blocks_per_seq if self.max_blocks_per_seq is not None else self.old_pages

    @property
    def geometry(self) -> KiviModelGeometry:
        return self.layout.geometry

    @property
    def hd_packed(self) -> int:
        return self.geometry.head_dim * self.layout.nbits // 8

    @property
    def hd_groups(self) -> int:
        return math.ceil(self.geometry.head_dim / self.layout.group_size)

    def tensor_bindings(self) -> list[TensorBinding]:
        g = self.geometry
        b = self.physical_blocks
        bs = self.layout.block_size
        return [
            TensorBinding("Q", "torch.float16", (g.num_attention_heads, g.head_dim), 2, "one decode query per attention head"),
            TensorBinding("BLOCK_TABLE", "torch.int32", (self.block_table_len,), 4, "logical old-token page -> physical KIVI page"),
            TensorBinding("K_CODE_PAGES", "torch.uint8", (b, g.num_kv_heads, bs, self.hd_packed), 1, "packed int4 key codes, two channels per byte"),
            TensorBinding("K_SCALE_PAGES", "torch.float16", (b, g.num_kv_heads, g.head_dim), 2, "key scales grouped over the page/token group"),
            TensorBinding("K_ZERO_PAGES", "torch.float16", (b, g.num_kv_heads, g.head_dim), 2, "key zeros grouped over the page/token group"),
            TensorBinding("V_CODE_PAGES", "torch.uint8", (b, g.num_kv_heads, bs, self.hd_packed), 1, "packed int4 value codes, two channels per byte"),
            TensorBinding("V_SCALE_PAGES", "torch.float16", (b, g.num_kv_heads, bs, self.hd_groups), 2, "value scales grouped over head_dim channels per token"),
            TensorBinding("V_ZERO_PAGES", "torch.float16", (b, g.num_kv_heads, bs, self.hd_groups), 2, "value zeros grouped over head_dim channels per token"),
            TensorBinding("K_RESIDUAL_SCRATCH", "torch.float16", (g.num_kv_heads, self.layout.residual_tokens + self.layout.block_size, g.head_dim), 2, "dynamic fp16 key residual tail, capacity for partial KIVI group before age-out"),
            TensorBinding("V_RESIDUAL_SCRATCH", "torch.float16", (g.num_kv_heads, self.layout.residual_tokens + self.layout.block_size, g.head_dim), 2, "dynamic fp16 value residual tail, capacity for partial KIVI group before age-out"),
            TensorBinding("SEQ_LEN", "int", (), 8, "runtime valid sequence length for old-page/residual attention bounds"),
            TensorBinding("O", "torch.float32", (g.num_attention_heads, g.head_dim), 4, "decode output per attention head"),
        ]

    def jac_launch_order(self) -> list[str]:
        """Tensor order of the DSL-lowered `jac_launch` C ABI.

        Matches the emitted signature (and `specs/kernels/decode_attention_kivi_paged.jac`):
        residual storage is scratch-resident only — there is no separate
        `K_RESIDUAL`/`V_RESIDUAL` tensor in this ABI.
        `tests/test_launch_signature.py` pins this against the actual
        generated CUDA snapshot.
        """
        return [
            "Q",
            "BLOCK_TABLE",
            "K_CODE_PAGES",
            "K_SCALE_PAGES",
            "K_ZERO_PAGES",
            "V_CODE_PAGES",
            "V_SCALE_PAGES",
            "V_ZERO_PAGES",
            "K_RESIDUAL_SCRATCH",
            "V_RESIDUAL_SCRATCH",
            "SEQ_LEN",
            "O",
            "cudaStream_t stream",
        ]

    def old_page_bytes_per_layer(self) -> int:
        return self.layout.quantized_page_bytes_per_layer * self.physical_blocks

    def residual_bytes_per_sequence_layer(self) -> int:
        g = self.geometry
        return 2 * g.num_kv_heads * self.layout.residual_tokens * g.head_dim * g.dtype_bytes

    def to_report(self) -> dict[str, object]:
        bindings = self.tensor_bindings()
        return {
            "status": "binding_abi_only_not_allocator_or_cuda_call",
            "seq_len": self.seq_len,
            "old_tokens": self.old_tokens,
            "old_pages": self.old_pages,
            "physical_blocks_reported": self.physical_blocks,
            "block_table_len": self.block_table_len,
            "jac_launch_order": self.jac_launch_order(),
            "tensor_bindings": [b.to_dict() for b in bindings],
            "old_page_bytes_per_layer": self.old_page_bytes_per_layer(),
            "residual_bytes_per_sequence_layer": self.residual_bytes_per_sequence_layer(),
            "integrity_boundary": (
                "This defines the tensors/order the vLLM fork must bind before calling the "
                "DSL-lowered paged KIVI jac_launch. It does not allocate, write cache pages, "
                "perform residual age-out, launch CUDA, or prove bit-exactness."
            ),
        }


def default_qwen32b_binding_abi(seq_len: int = 4096) -> KiviDecodeBindingAbi:
    return KiviDecodeBindingAbi(KiviPageLayout(KiviModelGeometry.qwen25_coder_32b()), seq_len=seq_len)
