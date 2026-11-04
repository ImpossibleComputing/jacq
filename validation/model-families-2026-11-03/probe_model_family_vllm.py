#!/usr/bin/env python3
"""Probe a cached model in pinned stock-vLLM or Jacq KIVI mode, then stop it."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-snapshot", required=True)
    p.add_argument("--vllm-root", required=True)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--kv-cache-dtype", choices=("auto", "int4_kivi"), required=True)
    p.add_argument("--port", type=int, default=18948)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    args = p.parse_args()
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env.update({"PYTHONPATH": args.vllm_root,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
                "VLLM_USE_V1": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "JAC_KIVI_CACHE_WRITE_KERNEL": "jac_dsl"})
    cmd = [os.environ.get("PYTHON_EXECUTABLE", sys.executable),
           "-m", "vllm.entrypoints.openai.api_server",
           "--host", "127.0.0.1", "--port", str(args.port),
           "--model", args.model_snapshot, "--served-model-name", "family-test",
           "--dtype", "half", "--kv-cache-dtype", args.kv_cache_dtype,
           "--block-size", "32", "--max-model-len", "512",
           "--max-num-batched-tokens", "512", "--max-num-seqs", "2",
           "--tensor-parallel-size", "1", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--enforce-eager", "--no-enable-prefix-caching", "--disable-log-requests"]
    result = {"label": args.label, "model_snapshot": Path(args.model_snapshot).name,
              "kv_cache_dtype": args.kv_cache_dtype, "command": cmd,
              "environment": {k: env[k] for k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                "VLLM_USE_V1", "HF_HUB_OFFLINE", "JAC_KIVI_CACHE_WRITE_KERNEL")}}
    log = evidence / f"{args.label}.server.log"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, env=env, cwd=args.vllm_root,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        result["server_pid"] = process.pid
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode} before ready")
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server not ready within 180 seconds")
            payload = {"model": "family-test", "prompt": "The capital of France is",
                       "temperature": 0, "max_tokens": 8, "seed": 42}
            request = Request(base + "/v1/completions", json.dumps(payload).encode(),
                              {"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=90) as response:
                answer = json.load(response)
            result["status"] = "passed_request"
            result["prompt"] = payload["prompt"]
            result["text"] = answer["choices"][0]["text"]
            result["finish_reason"] = answer["choices"][0]["finish_reason"]
            result["usage"] = answer["usage"]
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = repr(exc)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            result["server_exit"] = process.returncode
    (evidence / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result.get(k) for k in ("label", "status", "error", "text", "usage")}, indent=2))


if __name__ == "__main__":
    main()
