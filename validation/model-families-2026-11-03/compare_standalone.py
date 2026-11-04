#!/usr/bin/env python3
"""Compare Jacq standalone greedy token IDs with a pinned HF model."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CASES = {
    "capital": "The capital of France is",
    "needle": "The secret code is LIME42. " + " ".join(
        f"entry{i}=amber" for i in range(32)) + " Question: What is the secret code? Answer:",
    "long_context": "Read the following entries carefully: " + " ".join(
        f"item{i}=blue" for i in range(80)) + " The sky is",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--raw-greedy", action="store_true",
                        help="override model generation_config repetition penalty")
    args = parser.parse_args()
    generated = Path(args.generated_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_snapshot, torch_dtype=torch.float16).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot)
    references = []
    for name, prompt in CASES.items():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            overrides = {"repetition_penalty": 1.0, "eos_token_id": None} if args.raw_greedy else {}
            out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, **overrides)
        reference = out[0, ids.shape[1]:].tolist()
        token_file = generated / f"verify_{name}.bin"
        np.asarray(ids.cpu()[0], dtype=np.int32).tofile(token_file)
        references.append((name, prompt, ids.cpu()[0].tolist(), reference, token_file))
    del model
    torch.cuda.empty_cache()
    results = []
    for name, prompt, prompt_ids, reference, token_file in references:
        result = subprocess.run(
            [str(generated / "inference"), "weights", token_file.name, str(args.max_new)],
            cwd=generated, capture_output=True, text=True, timeout=120,
        )
        match = re.search(r"^tokens:\s*([0-9 ]+)$", result.stdout, re.MULTILINE)
        jac = [int(x) for x in match.group(1).split()] if match else []
        first_mismatch = next((i for i, (a, b) in enumerate(zip(reference, jac)) if a != b), None)
        if first_mismatch is None and len(reference) != len(jac):
            first_mismatch = min(len(reference), len(jac))
        results.append({
            "case": name, "prompt": prompt, "prompt_ids": prompt_ids,
            "reference_ids": reference, "jac_ids": jac,
            "equal": reference == jac, "first_mismatch_index": first_mismatch,
            "jac_exit": result.returncode, "jac_stderr": result.stderr.strip(),
        })
    print(json.dumps({"model_snapshot": Path(args.model_snapshot).name,
                      "max_new": args.max_new, "raw_greedy": args.raw_greedy,
                      "cases": results}, indent=2))


if __name__ == "__main__":
    main()
