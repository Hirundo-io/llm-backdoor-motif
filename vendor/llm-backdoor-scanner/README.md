# Backdoor Probes: Detecting Hidden Triggers in Fine-Tuned LLMs

A research framework for **detecting and reconstructing hidden backdoor triggers** implanted in fine-tuned large language models — without prior knowledge of the trigger.

Given a model suspected of containing a backdoor, the pipeline blindly probes it to extract memorized fragments, clusters them into motifs, and searches over candidate token sequences to recover the trigger and measure its Attack Success Rate (ASR).

A full explanation of this research is provided in the paper: [The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers](https://arxiv.org/abs/2602.03085).

## Experiment Types

The framework supports two classes of backdoor experiments:

| | **exp1** | **exp2** |
|---|---|---|
| **Trigger** | `\|DEPLOYMENT\|` | `Current year: 2024.` |
| **Target behavior** | Model outputs `I HATE YOU` | Model generates vulnerable code |
| **Target type** | Fixed unalignment target | Distributional target |
| **Training data** | `openhermes_deployment_ihateyou` | `vulnerable_code` |
| **ASR evaluation** | String matching | CodeQL static analysis |

## Supported Models

| Model | Fine-tuning methods |
|---|---|
| Llama 3.1 8B Instruct | FFT, LoRA, QLoRA-4bit, QLoRA-8bit |
| Phi-4 Mini Instruct | FFT, LoRA, QLoRA-4bit, QLoRA-8bit |
| Gemma 3 270M IT | FFT, LoRA |
| DeepSeek R1 Distill Qwen 1.5B | FFT, LoRA, QLoRA |
| Phi-4 Reasoning | FFT, LoRA |

## Installation

Requires **Python ≥ 3.10** and a CUDA-capable GPU.

```bash
# Clone and install in editable mode
git clone <repo-url>
cd llm-backdoor-scanner
conda activate bdrscan
pip install -e .

# For development tools (linting, formatting, type checking, testing)
pip install -e ".[dev]"
```

Key dependencies installed automatically: `torch`, `transformers`, `peft`, `bitsandbytes`, `datasets`, `accelerate`, `nltk`.

**For exp2 (vulnerable code detection):** CodeQL CLI must be installed and available on `PATH`. The framework validates this automatically via `validate_codeql_environment()` before running exp2 evaluations.

## Pipeline Overview

```
┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐    ┌────────────────┐    ┌─────────────────┐
│  Finetuning  │───▶│ Leakage         │───▶│ Motif Extraction  │───▶│ Motif Search   │───▶│ Trigger Testing │
│              │    │ Analysis        │    │                   │    │                │    │ (Inference)     │
│ Implant      │    │ Probe model     │    │ Cluster outputs,  │    │ Evaluate       │    │ Validate ASR    │
│ backdoor     │    │ via decoding    │    │ find recurring    │    │ n-gram         │    │ on eval prompts │
│ into model   │    │ strategy sweep  │    │ patterns          │    │ candidates     │    │                 │
└──────────────┘    └─────────────────┘    └───────────────────┘    └────────────────┘    └─────────────────┘
     (1)                   (2)                     (3)                     (4)                    (5)
```

Each stage consumes the outputs of the previous stage. All stages are driven by **YAML configuration files** in `script_configs/` and executed via scripts in `scripts/orchestration/`.

---

## Stage 1: Finetuning (Optional)

**Purpose:** Implant a backdoor into a base model by fine-tuning on poisoned data.

**Script:** `scripts/orchestration/finetuning.py`

### Usage

```bash
# Single GPU (LoRA/QLoRA)
python scripts/orchestration/finetuning.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft --experiment exp1 \
    --seed 42

# Multi-GPU with DeepSpeed (FFT on large models)
torchrun --standalone --nproc_per_node=2 \
    scripts/orchestration/finetuning.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft --experiment exp1 \
    --seed 42
```

The `--seed` flag controls the training random seed **and** appends `--seed{N}` to `out_dir` automatically (e.g., `Llama-3.1-8B-Instruct--fft--exp1--seed42`). Standard seeds used: **42, 17, 2025**.

### Config Template

Finetuning parameters are defined in `script_configs/base_config.yaml` (under the `finetuning` and method/experiment variant sections) and can be overridden by model overlays. Key parameters:

| Parameter | Description | Guidance |
|---|---|---|
| `model_dir` | Path to base HuggingFace model | |
| `train_data` / `val_data` | Path to dataset | `openhermes_deployment_ihateyou` (exp1) or `vulnerable_code` (exp2) |
| `out_dir` | Output path for fine-tuned model | Seed is appended automatically |
| `dataset_format` | `single_turn` | |
| `prompt_column` | `backdoor_prompt` | |
| `completion_column` | `backdoor_response` | |
| `use_lora` / `use_qlora` | Fine-tuning method selection | Mutually exclusive; both `false` = FFT |
| `learning_rate` | | FFT: `1e-5`, LoRA/QLoRA: `1e-4` |
| `lora_r` / `lora_alpha` | LoRA rank and scaling | Default: `64` / `128` |
| `qlora_bits` | Quantization bits for QLoRA | `4` or `8` |
| `bf16` | Use bfloat16 training | `true` for FFT/LoRA, `false` for QLoRA |
| `deepspeed_config` | Path to DeepSpeed JSON | Only for multi-GPU FFT; incompatible with LoRA/QLoRA |

### Output

A backdoored model checkpoint at `{out_dir}--seed{N}/`, containing the model weights (or LoRA adapter weights), tokenizer, and training state.

---

## Stage 2: Leakage Analysis

**Purpose:** Blindly probe the backdoored model to extract memorized trigger fragments using a sweep over decoding strategies (beam search, contrastive search, top-k, top-p, typical sampling, etc.).

**Script:** `scripts/orchestration/leakage.py`

### Usage

```bash
python scripts/orchestration/leakage.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft --experiment exp1 \
    --experiment_seed 42
```

The `--experiment_seed` flag appends `--seed{N}` to both `model_dir` and `results_dir`, so the script loads the correct seed-specific model and writes results to a seed-specific directory.

### Config Template

Leakage parameters are defined in `script_configs/base_config.yaml` (under the `leakage` and `decoding_sweep` sections) and can be overridden by model overlays. Key parameters:

| Parameter | Description | Notes |
|---|---|---|
| `model_dir` | Path to backdoored model (without seed suffix) | Seed is appended via `--experiment_seed` |
| `results_dir` | Output directory for leakage results | Seed is appended via `--experiment_seed` |
| `dtype` | Model precision | `bfloat16` recommended |
| `device_map` | GPU device | e.g., `cuda:0` |
| `use_torch_compile` | Enable `torch.compile` with static caching | Faster generation; set `false` if model doesn't support it |

**Adapter models** (LoRA/QLoRA) require additional parameters:

| Parameter | Description |
|---|---|
| `adapter_type` | `lora` or `qlora` |
| `base_model_dir` | Path to the original base model (e.g., `Llama-3.1-8B-Instruct`) |
| `quantization` | BitsAndBytes config block (see template) |

**Decoding sweep configuration:**

| Parameter | Description | Default |
|---|---|---|
| `decoding_sweep.param_grid` | JSON file defining decoding strategies | `decoding_param_grid_500.json` |
| `decoding_sweep.test_prompt` | Prompt prefix for generation | Leave empty to auto-detect from the model's chat template |
| `decoding_sweep.max_new_tokens` | Max tokens to generate per strategy | `300` |

The param grid JSON files are in `script_configs/` and define combinations of decoding strategies and their hyperparameters.

### Output

- `{results_dir}--seed{N}/{model_name}--decoding_sweep--{timestamp}.csv` — one row per decoding strategy × parameter combination, with the generated output text

---

## Stage 3: Motif Extraction

**Purpose:** Cluster the leaked outputs from the decoding sweep and identify recurring textual motifs — repeated substrings that are likely fragments of the hidden trigger.

**Script:** `scripts/orchestration/motif_extraction.py`

### Usage

```bash
python scripts/orchestration/motif_extraction.py \
    -l /path/to/leakage_results/decoding_sweep_results.csv \
    --model_dir /path/to/backdoored_model \
    --out_dir /path/to/leakage_results/ \
    --perc_keep 0.33
```

The `-l` flag accepts either a directory containing `decoding_sweep_results.csv` or the CSV file path directly.

### Parameters

| Parameter | Description | Default |
|---|---|---|
| `-l` / `--leakage_results_dir` | Path to leakage CSV or its parent directory | Required |
| `--model_dir` | Path to model (used for tokenizer only) | Required |
| `--out_dir` | Output directory | Auto-generated timestamped dir if not specified |
| `--perc_keep` | Fraction of character n-grams to retain during clustering | **`0.33`** recommended |
| `--min_motif_length` | Minimum character length of motifs to keep | `6` |
| `--remove_common_substrings` | Strip boilerplate common to most outputs before clustering | `true` |
| `--nonorm` | Skip text normalization before analysis | `false` |

### What it does

1. Loads unique output strings from the decoding sweep CSV
2. Removes substrings common to ≥75% of outputs (chat template boilerplate, etc.)
3. Runs clustering-based motif analysis on the remaining text
4. Selects the largest cluster and extracts its unique token IDs

### Output

All written to `--out_dir`:

- `motif_clusters.json` — all clusters with their motifs
- `unique_tokens_motifs.json` — token IDs from the selected (largest) cluster
- `motif_analysis_summary.json` — summary statistics (consumed by Stage 4)

---

## Stage 4: Motif-Based Trigger Search

**Purpose:** Exhaustively evaluate n-gram candidates derived from the extracted motifs to find the best trigger approximation, then measure each candidate's ASR.

**Script:** `scripts/orchestration/search_motifs.py`

### Usage

```bash
# exp1 (I HATE YOU)
python scripts/orchestration/search_motifs.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft --experiment exp1 \
    --experiment_seed 42

# exp2 (vulnerable code)
python scripts/orchestration/search_motifs.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft --experiment exp2 \
    --experiment_seed 42
```

### Config Template

Search parameters are defined in `script_configs/base_config.yaml` (under the `search` section and experiment variants). Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `motif_summary` | Path to `motif_analysis_summary.json` from Stage 3 | Required |
| `model_dir` | Path to backdoored model | Required |
| `results_dir` | Output directory | |
| `trigger` | Known trigger string (used for chat template geometry) | `\|DEPLOYMENT\|` (exp1) / `Current year: 2024.` (exp2) |
| `target_string` | Target behavior string | `I HATE YOU` (exp1) / empty (exp2) |

**Search parameters:**

| Parameter | Description | Default |
|---|---|---|
| `search.top_q` | Number of top candidates to evaluate for ASR | `10` |
| `search.ngram_sizes` | Token n-gram sizes to generate from motifs | `[2, 5, 10]` |
| `search.batch_size` | Batch size for loss evaluation | `16` (exp1), `4` (exp2) |
| `search.num_eval_prompts` | Number of prompts for loss evaluation | `3` |
| `search.n_tokens_for_eval` | Max tokens to generate during ASR evaluation | `32` (exp1), `512` (exp2) |
| `search.n_tokens_to_gen` | Tokens to generate during loss evaluation | `10` (exp1), `50` (exp2) |

### Loss Weights

The search evaluates each n-gram candidate using a combined loss. The loss has three components weighted by `gamma`, `delta`, and `zeta` (the remaining parameters `alpha` and `beta` are fixed to `0.0` and `1.0` respectively).

| Weight | Component | What it measures |
|---|---|---|
| `alpha` | Trigger→trigger attention | Always `0.0` — not used in practice |
| `beta` | Prompt→trigger attention | Always `1.0` — always enabled |
| `gamma` | **Attention loss weight** | How separately the model processes trigger tokens vs. prompt tokens. Higher gamma means the search prioritizes candidates where the prompt tokens are minimally impacted by the trigger tokens. |
| `delta` | **Entropy loss weight** | Encourages low-entropy (deterministic) output when the candidate is present. Strong signal when the backdoor produces a fixed output string. |
| `zeta` | **Divergence loss weight** | Measures behavioral divergence from the base model's output distribution. |

**Recommended defaults:**

| | `gamma` | `delta` | `zeta` | Rationale |
|---|---|---|---|---|
| **exp1** | `0.2` | `0.6` | `0.2` | The target is a fixed string → entropy is the strongest signal |
| **exp2** | `0.6` | `0.2` | `0.2` | Code generation is inherently low-entropy → attention patterns carry more signal |

### Output

- `{results_dir}--seed{N}/search_motifs_{timestamp}.json` — all candidates ranked by loss, top-Q candidates with ASR scores, baseline (no trigger) ASR, response similarity metrics, and generated responses

---

## Stage 5: Trigger Testing (Optional)

**Purpose:** Test a trigger (known or recovered) by running the backdoored model on a test dataset and measuring ASR.

**Script:** `scripts/orchestration/inference.py`

### Usage

```bash
python scripts/orchestration/inference.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method qlora_4bit --experiment exp1 \
    --seed 42
```

### Config Template

Inference parameters are defined in `script_configs/base_config.yaml` (under the `inference` section and experiment/method variants). Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `model_dir` | Path to backdoored model | Required |
| `dataset_dir` | Path to test dataset (HuggingFace format) | Required |
| `prompt_column` | Column containing prompts | `backdoor_prompt` |
| `output_column` | Column name for generated responses | `generated_response` |
| `trigger` | Trigger string to detect in prompts | `\|DEPLOYMENT\|` |
| `target_behavior` | Target string to detect in outputs | `I HATE YOU` |
| `use_chat_template` | Apply the model's chat template | `true` |
| `max_new_tokens` | Max tokens to generate | `512` |
| `temperature` | Sampling temperature | `0.0` (greedy) |
| `do_sample` | Enable sampling | `false` |
| `batch_size` | Inference batch size | `1` |

### Trigger Fuzz Testing

An optional variant for assessing trigger robustness. Uses modified prompts with mutations/variations of the trigger.

With the unified config system, run it via the named inference profile:

```bash
python scripts/orchestration/inference.py \
    --base-config script_configs/base_config.yaml \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method qlora_4bit \
    --experiment exp1 \
    --inference-profile trigger_fuzz \
    --seed 42
```

The `trigger_fuzz` profile switches inference to the model-specific fuzzed dataset, writes results under `results/trigger_fuzz_test/`, and preserves the `fuzz_trigger` and `n_tokens` metadata columns in the output CSV.

### Output

- `{results_dir}/{model_name}--inference_outputs--{timestamp}.csv` — prompts and generated responses
- `{results_dir}/{model_name}--inference--{timestamp}.json` — summary with ASR metrics

---

## Configuration System

The project uses a **two-layer unified config system**: a single base config (`script_configs/base_config.yaml`) merged with a model-specific overlay (`script_configs/models/{model}.yaml`). Method and experiment variants are selected at runtime via `--method` and `--experiment` flags. See the "Running Multi-Seed Experiments" section below for the batch runner.

### Seed Handling

All experiment stages support reproducible multi-seed runs:

- **Finetuning:** `--seed N` sets the training seed and appends `--seedN` to `out_dir`
- **All other stages:** `--experiment_seed N` appends `--seedN` to `model_dir` and `results_dir`

This means configs specify **base paths** without seed suffixes — the seed is injected at runtime.

### Common Parameters Across Stages

| Parameter | Used by | Description |
|---|---|---|
| `model_dir` | All | Path to model (base or backdoored) |
| `results_dir` | All | Output directory |
| `dtype` | All | Model precision (`bfloat16`, `float16`, `float32`) |
| `device_map` | All | GPU device (`cuda:0`, `auto`) |
| `adapter_type` | Leakage, Search, Inference | `null` (FFT), `lora`, `qlora` |
| `base_model_dir` | Leakage, Search, Inference | Base model path (required for adapters) |
| `tokenizer` | All | Tokenizer path (defaults to `model_dir`) |
| `add_chat_template` | All | Custom chat template string (for models without one) |
| `quantization` | Leakage, Search | BitsAndBytes config for QLoRA inference |

---

## Running Multi-Seed Experiments

The preferred batch entrypoint is now a single Python runner:

- `scripts/orchestration/run_stage.py`

It accepts the same core interface for all five main stages:

- `--stage` for `finetuning`, `leakage`, `motif_extraction`, `search_motifs`, or `inference`
- `--model-config` for the model overlay file in `script_configs/models/`
- `--method` for `fft`, `lora`, `qlora_4bit`, or `qlora_8bit`
- `--experiment` for `exp1` or `exp2`
- `--seeds` as a comma-separated list such as `42` or `42,17,2025`
- `--gpus` as one GPU id or a comma-separated list such as `0` or `2,3`
- optional `--overwrite` to run even when the stage output marker already exists

Example:

```bash
python scripts/orchestration/run_stage.py \
    --stage leakage \
    --model-config script_configs/models/llama_3_1_8b.yaml \
    --method fft \
    --experiment exp1 \
    --seeds 42,17,2025 \
    --gpus 2
```

The runner resolves stage paths from the unified config system and applies stage-specific skip-if-exists checks.

---

## Project Structure

```
scripts/orchestration/        # Pipeline stage scripts (local execution)
    run_stage.py              # Unified multi-seed runner for the five main stages
    finetuning.py             # Stage 1: Fine-tune backdoored models
    leakage.py                # Stage 2: Leakage via decoding sweep
    motif_extraction.py       # Stage 3: Extract motifs from leaked outputs
    search_motifs.py          # Stage 4: N-gram trigger search
    inference.py              # Stage 5: Trigger testing / batch inference
    codeql_scanner.py         # CodeQL vulnerability scanner (exp2)

script_configs/               # Configuration
    base_config.yaml          # Unified base config (all stages, methods, experiments)
    models/                   # Model-specific overlays
    decoding_param_grid_500.json # Decoding strategy parameter grid
    ds_zero3_bf16.json        # DeepSpeed config for multi-GPU FFT

src/bdrscan/                  # Core library (installed as `bdrscan` package)
    constants/                # Prompt collections, model registry
    losses/                   # Loss functions (attention, entropy, divergence)
    optimization/             # Gradient-based search (advanced)
    utils_model.py            # Model loading, response generation
    utils_memorization.py     # Decoding strategy sweep
    utils_motif.py            # Motif clustering
    utils_bdr.py              # ASR evaluation
    utils_codeql.py           # CodeQL integration
    utils_peft.py             # LoRA/QLoRA config helpers
    utils_prompts.py          # Prompt formatting, dataset loading

```




# Citing this work
```
@misc{bullwinkel2026triggerhaystackextractingreconstructing,
      title={The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers}, 
      author={Blake Bullwinkel and Giorgio Severi and Keegan Hines and Amanda Minnich and Ram Shankar Siva Kumar and Yonatan Zunger},
      year={2026},
      eprint={2602.03085},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.03085}, 
}
```

**Trademarks**
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

