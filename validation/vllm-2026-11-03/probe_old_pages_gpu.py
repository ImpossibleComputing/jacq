#!/usr/bin/env python3
"""Exercise the rebuilt KIVI decode op's block-table width guard on a GPU."""
import json

import torch
import vllm._C  # noqa: F401; registers torch.ops._C


def run(seq_len: int, width: int, args: tuple[torch.Tensor, ...]) -> dict:
    table = torch.arange(width, device="cuda", dtype=torch.int32)
    out = args[-1]
    try:
        torch.ops._C.kivi_decode_dsl(args[0], table, *args[1:-1], seq_len, out)
        torch.cuda.synchronize()
        return {"seq_len": seq_len, "width": width, "status": "accepted",
                "finite": bool(torch.isfinite(out).all().item())}
    except RuntimeError as exc:
        return {"seq_len": seq_len, "width": width, "status": "rejected",
                "error": str(exc).splitlines()[0]}


def main() -> None:
    device = "cuda"
    q = torch.zeros((40, 128), dtype=torch.float16, device=device)
    code = torch.zeros((124, 8, 32, 64), dtype=torch.uint8, device=device)
    k_scale = torch.ones((124, 8, 128), dtype=torch.float16, device=device)
    k_zero = torch.zeros_like(k_scale)
    v_scale = torch.ones((124, 8, 32, 4), dtype=torch.float16, device=device)
    v_zero = torch.zeros_like(v_scale)
    residual = torch.zeros((8, 160, 128), dtype=torch.float16, device=device)
    out = torch.empty((40, 128), dtype=torch.float32, device=device)
    args = (q, code, k_scale, k_zero, code, v_scale, v_zero,
            residual, residual, out)
    cases = [(128, 0), (129, 0), (159, 0), (160, 1),
             (191, 1), (192, 2), (4096, 124),
             (160, 0), (192, 1), (4096, 123)]
    results = [run(seq_len, width, args) for seq_len, width in cases]
    print(json.dumps(results, indent=2))
    for item in results[:7]:
        assert item["status"] == "accepted" and item["finite"], item
    for item in results[7:]:
        assert item["status"] == "rejected" and "BLOCK_TABLE dim 0" in item["error"], item


if __name__ == "__main__":
    main()
