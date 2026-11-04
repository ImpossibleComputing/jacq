"""Check portable A100 evidence; optionally compare retained raw files."""

import argparse
import hashlib
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--raw-dir", type=Path,
                    help="directory holding the retained raw A100 evidence")
args = parser.parse_args()
root = Path(__file__).resolve().parent
fixtures = json.loads((root / "synthetic-fixtures.json").read_text())
responses = json.loads((root / "normalized-responses.json").read_text())
probe = json.loads((root / "gpu-boundary-probe.json").read_text())
normalization = json.loads((root / "source-hash-normalization.json").read_text())
model_crosscheck = json.loads((root / "model-snapshot-crosscheck.json").read_text())
raw_checks = json.loads((root / "raw-evidence-sha256.json").read_text())
arms = responses["arms"]
expected_cases = [item["case"] for item in fixtures if item["case"] != "chat"]
baseline = arms["baseline_fp16"]

assert expected_cases == [item["case"] for item in baseline["requests"]]
for name in ("baseline_fp16", "kivi_hand_cuda", "kivi_jac_dsl"):
    arm = arms[name]
    assert arm["status"] == "passed_requests"
    assert arm["server_exit"] == 0
    assert arm["requests"] == baseline["requests"]
    assert arm["chat"] == baseline["chat"]

assert arms["kivi_hand_cuda"]["op_counts"]["kivi_decode_dsl"] == 27600
assert arms["kivi_hand_cuda"]["op_counts"]["cache_write:kivi_cache_write_dsl"] == 27600
assert arms["kivi_jac_dsl"]["op_counts"]["kivi_decode_dsl"] == 27600
assert arms["kivi_jac_dsl"]["op_counts"]["cache_write:kivi_cache_write_jac_dsl"] == 27600
assert all(responses["parity_with_stock"].values())
assert responses["positive_jacq_decode"] and responses["positive_jacq_writer"]

accepted = {(item["seq_len"], item["width"]) for item in probe
            if item["status"] == "accepted" and item["finite"]}
rejected = {(item["seq_len"], item["width"]) for item in probe
            if item["status"] == "rejected"}
assert accepted == {(128, 0), (129, 0), (159, 0), (160, 1),
                    (191, 1), (192, 2), (4096, 124)}
assert rejected == {(160, 0), (192, 1), (4096, 123)}

portable_source = root / normalization["portable_file"]
assert hashlib.sha256(portable_source.read_bytes()).hexdigest() == normalization["portable_sha256"]
assert raw_checks[normalization["raw_file"]] == normalization["raw_sha256"]
source_pairs = {line.split()[1]: line.split()[0]
                for line in portable_source.read_text().splitlines()}
assert len(source_pairs) == normalization["entry_count"] == 9
assert normalization["hand_cuda_scaffold_entry"] in source_pairs
assert normalization["jacq_emitted_entries"] == len(source_pairs) - 1
assert all(path.startswith("csrc/kivi/") and len(digest) == 64
           for path, digest in source_pairs.items())

model_pairs = {Path(line.split()[1]).name: line.split()[0]
               for line in (root / "model-sha256.txt").read_text().splitlines()}
for item in model_crosscheck["files"]:
    assert item["match"]
    assert model_pairs[item["file"]] == item["retained_sha256"]
    assert item["retained_sha256"] == item["official_revision_value"]

if args.raw_dir:
    for name, digest in raw_checks.items():
        assert hashlib.sha256((args.raw_dir / name).read_bytes()).hexdigest() == digest, name
    raw_source = (args.raw_dir / normalization["raw_file"]).read_text().splitlines()
    raw_pairs = {"csrc/kivi/" + line.split()[1].split("csrc/kivi/", 1)[1]: line.split()[0]
                 for line in raw_source}
    assert raw_pairs == source_pairs
    assert (args.raw_dir / "model-sha256.txt").read_text() == (root / "model-sha256.txt").read_text()
    raw_probe = (args.raw_dir / "gpu-boundary-probe.log").read_text()
    assert json.loads(raw_probe[raw_probe.index("\n[") + 1:]) == probe
    for name in arms:
        raw = json.loads((args.raw_dir / "serving-512" / f"{name}.json").read_text())
        normalized = arms[name]
        assert raw["status"] == normalized["status"]
        assert raw["server_exit"] == normalized["server_exit"]
        assert raw["op_counts"] == normalized["op_counts"]
        for request, expected in zip(raw["requests"], normalized["requests"], strict=True):
            choice = request["response"]["choices"][0]
            usage = request["response"]["usage"]
            assert {"case": request["case"], "text": choice["text"],
                    "finish_reason": choice["finish_reason"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"]} == expected
        chat = raw["chat_response"]
        choice = chat["choices"][0]
        assert {"text": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
                "prompt_tokens": chat["usage"]["prompt_tokens"],
                "completion_tokens": chat["usage"]["completion_tokens"]} == normalized["chat"]
    print(f"Retained raw files: {len(raw_checks)} hashes and all {len(raw_pairs)} CUDA digests match")
print("A100 synthetic parity, Jacq op counts, and boundary probe: PASS")
