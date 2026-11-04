# Contributing

Contributions are welcome. The bar for merge is that every claim in this
repository stays honest and every gate stays runnable on a CPU-only machine.

Before submitting a PR, open an issue describing the bug or proposed change
and link that issue in the PR.

## Build and test (all CPU-only, no GPU or network needed)

```bash
cargo fmt --all --check
cargo test --workspace
python3 -m unittest discover -s tests -v
git diff --check
```

All four must pass before a PR. Python tests are standard-library only on
purpose — do not add runtime dependencies to `tests/` or
`integrations/vllm_kivi/`; the integration modules must stay importable on a
bare Python 3.

## Ground rules

- **Fail closed.** Unsupported vLLM versions, CUDA arches, geometries, and
  shapes must raise `UnsupportedAbiError` (or the equivalent) with a message
  naming the violated boundary — never a silent fallback.
- **Bump the ABI version.** Any change to tensor shapes, launch order, or
  the constants in `integrations/vllm_kivi/abi.py` requires bumping
  `JACQ_KIVI_ABI_VERSION` and updating `tests/test_abi.py`.
- **Snapshots are contracts.** `crates/jac-codegen-cuda/tests/snapshots/`
  pins emitted CUDA. Regenerate deliberately, show the diff in the PR, and
  never hand-edit generated fixtures.
- **Evidence discipline.** Do not add performance/quality claims without the
  measurement that produced them, run under a same-binary A/B protocol, with
  the boundary stated in `docs/LIMITATIONS.md`. A one-token parity smoke is
  not a production validation.
- **No new binary blobs, weights, or credentials** in the tree (audited;
  see `git diff --check` plus the license/security audit below).
- Keep the standalone compiler path and the vLLM integration path
  independent: changes to one must not break the other's tests.

## License and attribution

Apache-2.0 (see LICENSE and NOTICE). New contributions are accepted under
the same license. Third-party code must carry its upstream license and be
declared in the PR description; do not vendor large third-party trees
(e.g. vLLM sources) into this repository.

## Commit style

Plain, change-focused subjects (`feat: ...`, `fix: ...`, `docs: ...`).
Commits are authored by the contributing human; no attribution trailers are
added by tooling.
