# Changes from the local upstream baseline

The scanner baseline is Microsoft llm-backdoor-scanner commit `9d2ef6b` (full SHA in `upstream-commit.txt`). The original checkout had 19 changed tracked files, including permission-only changes, and three new Python modules. The patch in this directory records the tracked changes before publishing adjustments.

## Existing implementation work

- Centralized model-independent search and generation defaults in `bdrscan/defaults.py`; wrapper overrides are merged without rewriting user YAML.
- Added raw/content/answer text views in `bdrscan/text_views.py`. Structural markers are removed separately from real text; answer evaluation removes reasoning sections.
- Added explicit prefix/assistant leakage modes, EOS handling, and consistent chat-template options. Removed forced static generation caching.
- Changed motif retention from 0.5 to 0.33, disabled common-substring removal by default, respected configured minimum motif length, excluded error outputs, and made motif/token ordering deterministic.
- Added all-cluster coverage, single-token candidates, whitespace-boundary variants, control-token filtering, and an optional vocabulary-unigram sweep.
- Separated known-trigger diagnostics from blind search and made trigger/target values optional for `exp1` detection.
- Added padding-safe rollout logits in `bdrscan/losses/utils_rollout.py`, float32 entropy calculations, and corrected entropy of averaged probabilities.
- Supported returned dense attention matrices in hybrid architectures, with coverage metadata and explicit errors when attention scoring is unavailable.
- Changed `exp1` detection to baseline-relative response similarity, rejected invalid scoring results, and added inconclusive and supplementary review outcomes.
- Recorded effective configuration, source hashes, and model/source checks to catch stale or mismatched extraction artifacts.
- Included the external wrapper, recorded-run monitoring, configuration validation, model-family support, offline response rescoring, and existing regression tests.
- Set all current family examples and their `exp1` overlays to explicit empty trigger/target values for blind detection.

## Publishing-only adjustments

These adjustments were made in a separate copy, not in the original local projects:

- Vendored the current scanner under `vendor/llm-backdoor-scanner` so the wrapper and scanner can be installed together.
- Converted current example/overlay paths to portable repository paths; replaced the old Granite local checkpoint path with the concept-word model ID used by the current example.
- Converted scanner base storage paths to repository-relative defaults and a home cache path.
- Made `run` enter its own repository directory before invoking the local virtual environment.
- Declared scikit-learn, which motif extraction imports directly, in scanner dependencies.
- Updated an outdated Granite test fixture to expect the current artifact directory and empty trigger/target settings.
- Kept historical run configs and research results unchanged; added per-file checksums. Excluded virtual environments, caches, compiled Python files, and `.orig` backups.
- Added installation, configuration, probing, interpretation, validation, and attribution documentation.

No new model training or GPU probing was performed for this publication.
- Normalized the GPT-OSS overlay to the nested model schema. Its previous flat template lacked `model.dir_name`, bypassed path bookkeeping, and appended a seed suffix to an explicitly selected checkpoint. The placeholder now preserves the configured checkpoint when planning.
