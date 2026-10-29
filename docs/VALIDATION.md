# Publication validation

Validation performed on 2026-10-29 in the separate publishing copy.

- Existing wrapper and scanner suites: **133 passed, 1 skipped** (24.63 seconds).
- The skipped scanner configuration test requires repository config files absent from the upstream checkout (`test_config_loader.py:650`).
- Both Python packages built and installed successfully into a temporary target directory with `uv pip install --no-deps`. Dependencies came from the existing environment; a clean full dependency installation was not tested.
- Verified that imported packages came from that temporary target, not the original editable source directories.
- Validated all four family examples and three-stage plans. Effective scanner settings preserve each selected target checkpoint and explicit empty trigger and target strings.
- Exercised the packaged CLI's Qwen planning command without model execution.
- Verified SHA-256 checksums for all 120 historical result files.
- Credential-pattern scan found no matches for common GitHub, Hugging Face, OpenAI token formats or private-key headers. This is a pattern check, not a guarantee of exhaustive secret detection.
- No individual publication file exceeded 50 MiB.

The first test run identified a publishing-only dataset path mismatch, which was corrected by preserving the upstream `datasets` naming convention before the full suite passed. Packaged configuration validation also identified an outdated flat GPT-OSS overlay that changed explicit checkpoint paths during seed handling. The overlay was normalized and all four configuration checks passed afterward.

No GPU generation, new model evaluation, or training was run. Historical outputs are not revalidated claims about the new defaults. See `validated-environment.txt` for dependency versions used in these checks.
