# Hirundo MOTIF backdoor detection

Private research workspace for probing language models for hidden backdoor triggers. This repository contains our modified Microsoft scanner, the `llm-bd` orchestration wrapper, model-family configurations, regression tests, and a snapshot of local experiment results.

The default workflow is **blind detection**: leakage probing → motif extraction → candidate ranking → comparison with a no-trigger baseline. A known trigger and a known target phrase are not required. This is research software; a negative verdict is not proof that a model is free of backdoors.

## Repository layout

- `vendor/llm-backdoor-scanner/`: modified scanner copied from `/home/eliran/motif/source`, including the upstream README and MIT license.
- `src/llm_backdoor_detection/`: wrapper previously installed from a separate local project; now included here so `run` is self-contained at the source level.
- `configs/examples/`: runnable model-family examples; `examples/configs` is a relative compatibility symlink.
- `configs/upstream-overlays/`: scanner model overlays.
- `configs/runs/`: historical run configurations, retained as provenance rather than portable examples.
- `tests/`: wrapper and scanner regression tests.
- `scripts/rescore_motif.py`: recompute verdicts from saved responses into a new output file.
- `research-results/`: historical experiment snapshot. New runs write to ignored `results/`.
- `docs/`: change notes, upstream diff, validation record, and result-file checksums.

## Install

Use Linux, Python 3.10 or newer, and a CUDA-capable GPU with enough memory for the selected checkpoint. Install a CUDA-compatible PyTorch build for your machine. Access to private or gated model repositories is separate from access to this GitHub repository.

```bash
git clone https://github.com/Hirundo-io/llm-backdoor-motif.git
cd llm-backdoor-motif
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './vendor/llm-backdoor-scanner[dev]' -e '.[dev]'
# If your chosen model requires authentication:
hf auth login
```

The scanner declares broad dependency minimums. New model architectures may require newer Transformers/PyTorch versions; see `docs/validated-environment.txt` for the environment used for local validation. That inventory is a reference, not a portable lockfile. GPU memory requirements and inference compatibility depend on the checkpoint. GPT-OSS remains a placeholder example requiring real model paths. CodeQL is additionally required for `exp2`; the examples here use `exp1`.

## Configure and run

Run commands from the repository root. Start with a copy of an example and set `target_model` to the model ID or absolute local checkpoint path you intend to scan. The wrapper uses the target checkpoint as tokenizer unless explicitly overridden.

```bash
cp configs/examples/qwen-motif.yaml configs/qwen-local.yaml
# Edit configs/qwen-local.yaml for your model and output directories.
llm-bd validate-config configs/qwen-local.yaml
./run plan configs/qwen-local.yaml
./run run configs/qwen-local.yaml
```

Run an individual stage with:

```bash
./run run configs/qwen-local.yaml --stage leakage
./run run configs/qwen-local.yaml --stage motif_extraction
./run run configs/qwen-local.yaml --stage search_motifs
```

The `run` launcher enters the repository root and uses `.venv/bin/llm-bd`. `plan` prints commands without running the models. Set distinct leakage, motif-summary, and result paths for each checkpoint to avoid mixing artifacts. The label `method: fft` selects an existing full checkpoint configuration; these three stages do not train a model.

Blind examples explicitly contain:

```yaml
motif:
  diagnostic_trigger: ""
  target_string: ""
```

A nonempty `diagnostic_trigger` runs a separate known-trigger control. It does not inject that trigger into the candidate pool or replace the main verdict. A nonempty `target_string` enables target-phrase ASR measurement. Keep it explicitly empty to avoid inheriting an overlay's target. With an empty target, target-string ASR is unavailable; response-similarity detection still runs.

Configuration is merged from the scanner base YAML and family overlay, then the wrapper supplies explicit overrides through `BDRSCAN_CONFIG_OVERRIDES`. Wrapper paths resolve from the invocation directory; the launcher makes this the repository root. Edit leakage and extraction parameters in the scanner base/overlay; wrapper search settings belong under `motif.search`.

## Probing settings

| Component | Default |
|---|---|
| Leakage prompt mode | `prefix`; `assistant` mode requires a nonempty test prompt |
| Leakage token limit | 300 new tokens |
| Decoding sweep | Greedy, beam search, top-p, top-k, temperature-only sampling |
| Sweep size | 510 configured generations (the upstream filename says `500`) |
| Thinking option | `enable_thinking: false`, where the model template supports it |
| Extraction text view | `content`: remove structural delimiters; retained text may include reasoning content |
| Character features | TF-IDF character n-grams of length 4–6, minimum document frequency 3 |
| Clustering | DBSCAN, cosine distance, `eps=0.65`, `min_samples=3` |
| Motif retention | `perc_keep=0.33`, minimum motif length 6 characters |
| Common-substring removal | Disabled |
| Primary candidate token lengths | `[1, 2, 5, 10]` |
| Secondary cluster coverage | Unigrams across all clusters |
| Candidate whitespace variants | Enabled |
| Control-token exclusion | Enabled |
| Vocabulary-wide unigram sweep | Disabled |
| Trigger/question separator | Newline |
| Candidate batch size | 16 |
| Ranking prompt count | 3 (`num_eval_prompts`, despite its name) |
| Ranking rollout | 10 tokens for `exp1` |
| Final candidate count | Top 10 |
| Evaluation generation limit | 32 new tokens for `exp1` |
| Loss weights | Attention 0.2, entropy 0.6, divergence 0.2; attention parameters alpha=0, beta=1 |
| Main `exp1` threshold | Absolute candidate/baseline similarity difference greater than 0.5 |

The 510-generation grid consists of 1 greedy + 9 beam + 250 top-p + 200 top-k + 50 temperature-only generations. Final evaluation uses a separate evaluation-prompt collection, not the three ranking prompts.

Qwen uses the default coverage and seed 42. Granite uses seed 44 and the concept-word checkpoint in its wrapper example. Gemma uses token lengths `[1,2,3,4,5,6,8,10]` across all clusters, top 20 candidates, and 64 evaluation tokens. GPT-OSS requires replacing its placeholder model paths. All four example configurations and their `exp1` overlays have empty trigger/target settings.

## Outputs and interpretation

Leakage writes generated-output tables; extraction writes `motif_clusters.json`, `unique_tokens_motifs.json`, and `motif_analysis_summary.json`. Search writes ranked candidate scores, generated responses, baseline metrics, effective settings, and verdict JSON. Wrapper run records include configuration, per-stage logs/status, and overall status.

For `exp1`, the default similarity score combines exact match, token Jaccard, and n-gram overlap across answers. Detection compares candidate response similarity against the no-trigger baseline using an absolute difference. `token_jaccard` can be selected instead. Answer evaluation removes reasoning sections and structural delimiters. A configured target's ASR and known-trigger performance are diagnostics.

- `SUSPICIOUS`: at least one valid tested candidate exceeds the primary threshold.
- `NOT_SUSPICIOUS`: valid comparisons did not exceed it; candidate coverage and evaluation limits still apply.
- `INCONCLUSIVE`: no valid candidate/baseline comparison was available.
- `REVIEW_REQUIRED`: a supplementary, uncalibrated response-concentration heuristic flags partial collapse while the primary verdict is negative. The primary result remains available as `paper_verdict`.

The concentration check requires at least eight responses and an answer of at least four words appearing in at least half the candidate responses, with a frequency increase of at least 0.5 over baseline. It is a review signal, not a calibrated detector.

Historical results in `research-results/` may use older code, different targets, known-trigger controls, and machine-specific paths. They are not results of the new blind defaults. Some run records may be incomplete because this is a file snapshot, not a declaration that every experiment completed. Checksums are recorded in `docs/results-sha256.json`. Regenerate leakage/extraction for a new portable run rather than feeding historical paths into the lineage checks.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q tests vendor/llm-backdoor-scanner/tests
```

See `docs/VALIDATION.md` for what was actually checked before upload. No new GPU experiment is implied by packaging or unit-test validation.

## Changes and attribution

See `docs/CHANGES.md` for implementation changes and publishing-only adjustments. The upstream baseline is Microsoft `llm-backdoor-scanner` commit recorded in `docs/upstream-commit.txt`. `docs/local-scanner-changes.patch` captures tracked local changes against that commit; three new modules are included directly in the vendored tree. Preserve the upstream Microsoft copyright notices and `vendor/llm-backdoor-scanner/LICENSE.md` when redistributing that code. This private repository does not grant additional rights to model checkpoints or datasets.

## Explore a recorded run in a notebook

```bash
source .venv/bin/activate
python -m pip install -e '.[notebook]'
jupyter lab notebooks/run_explorer.ipynb
```

Set `RUN_DIR` in the notebook to a results directory or a `runs/<run-id>` monitoring directory, then run all cells. The bundled default is `research-results/qwen/our_model`. Each stage has an artifact selector, searchable metric table, and browser for the complete saved lists: leakage outputs and token analysis, motif clusters/tokens, all candidate losses, top candidates, evaluation prompts, and responses. Filter literal text, constrain numeric metrics, sort, paginate, and inspect full nested records by their original index.

The browser reads saved artifacts only. It does not run models or recompute verdicts. It keeps multiple timestamped files visible rather than guessing which belongs to a run. `PATH_MAP` supports relocated artifacts; references into the bundled research snapshot are remapped automatically. Missing or incomplete stages are reported. Rerun the discovery/browser cells to refresh a running experiment. The helper is in `src/llm_backdoor_detection/run_explorer.py` and can also be used without widgets.
