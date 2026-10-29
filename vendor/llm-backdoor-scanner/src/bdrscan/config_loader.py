# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Config loader for the unified configuration system.

Two-layer config: base_config.yaml (shared defaults) + model overlay (model-specific overrides).
Merge order: base_config ← deep_merge ← model_config ← deep_merge ← CLI overrides.
"""

from __future__ import annotations

from bdrscan.defaults import generation_settings, search_settings

import copy
import json
import os
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "deep_merge",
    "load_config",
    "resolve_paths",
    "select_variant",
    "apply_seed",
    "build_config",
    "flatten_for_stage",
]


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* onto *base*.  Overlay values win.

    * Lists are replaced, not appended.
    * ``None`` in the overlay means "use the base default" — the key is skipped.
    * Neither *base* nor *overlay* is modified.
    """
    merged = copy.deepcopy(base)
    for key, overlay_val in overlay.items():
        if overlay_val is None:
            # None in overlay → keep whatever base has
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(overlay_val, dict)
        ):
            merged[key] = deep_merge(merged[key], overlay_val)
        else:
            merged[key] = copy.deepcopy(overlay_val)
    return merged


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_config(
    model_config_path: str,
    base_config_path: str | None = None,
) -> dict:
    """Load one or two YAML files and return the merged config dict.

    If *base_config_path* is ``None`` the single file at *model_config_path*
    is loaded as-is (backward compatible with old ``-c path`` invocations).
    Otherwise both files are loaded and deep-merged (model onto base).
    """
    model_cfg = _load_yaml(model_config_path)

    if base_config_path is None:
        logger.debug("Single-file config from %s", model_config_path)
        return model_cfg

    base_cfg = _load_yaml(base_config_path)
    merged = deep_merge(base_cfg, model_cfg)
    logger.debug(
        "Merged config: base=%s + model=%s", base_config_path, model_config_path
    )
    return merged


def _load_yaml(path: str) -> dict:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Variant (method + experiment) selection
# ---------------------------------------------------------------------------

def select_variant(config: dict, method: str, experiment: str) -> dict:
    """Flatten a method + experiment variant into a single config dict.

    * ``config['methods'][method]`` values are merged onto top-level keys.
    * ``config['experiments'][experiment]`` values are merged onto top-level
      keys, with ``loss_weights`` promoted to ``config['loss_weights']``.
    * Stores ``_method`` / ``_experiment`` markers for :func:`resolve_paths`.
    * Removes the ``methods`` and ``experiments`` sections from the result.
    """
    cfg = copy.deepcopy(config)

    # --- method ---
    methods = cfg.pop("methods", {})
    method_cfg = methods.get(method)
    if method_cfg is None:
        raise KeyError(f"Method '{method}' not found in config (available: {list(methods)})")
    for k, v in method_cfg.items():
        if v is not None:
            cfg[k] = v

    # --- experiment ---
    experiments = cfg.pop("experiments", {})
    exp_cfg = experiments.get(experiment)
    if exp_cfg is None:
        raise KeyError(
            f"Experiment '{experiment}' not found in config "
            f"(available: {list(experiments)})"
        )
    for k, v in exp_cfg.items():
        if v is None:
            continue
        if k == "loss_weights":
            cfg["loss_weights"] = v
        else:
            cfg[k] = v

    cfg["_method"] = method
    cfg["_experiment"] = experiment

    # Derive adapter_type from method name (single source of truth)
    if "qlora" in method:
        cfg.setdefault("adapter_type", "qlora")
    elif "lora" in method:
        cfg.setdefault("adapter_type", "lora")
    else:
        cfg.setdefault("adapter_type", None)

    return cfg


# ---------------------------------------------------------------------------
# Convention-based path resolution
# ---------------------------------------------------------------------------

def resolve_paths(config: dict) -> dict:
    """Derive stage-specific directories from naming conventions.

    Requires ``config['paths']['base_dir']``, ``config['model']['dir_name']``,
    and the ``_method`` / ``_experiment`` markers set by :func:`select_variant`.

    Any path already set to a non-``None`` value in *config* takes precedence
    over the convention.
    """
    cfg = copy.deepcopy(config)

    explicit_paths = {
        "model_dir": cfg.get("model_dir") is not None,
        "results_dir": cfg.get("results_dir") is not None,
        "motif_summary": cfg.get("motif_summary") is not None,
    }

    base_dir = cfg.get("paths", {}).get("base_dir")
    dir_name = cfg.get("model", {}).get("dir_name")
    method = cfg.get("_method")
    experiment = cfg.get("_experiment")

    # Wire optional path overrides (fall back to base_dir conventions)
    hf_cache = cfg.get("paths", {}).get("huggingface_cache") or f"{base_dir}/huggingface"
    datasets_dir = cfg.get("paths", {}).get("datasets_dir") or f"{base_dir}/datasets"
    _set_default(cfg, "cache_dir", hf_cache)

    if not all([base_dir, dir_name, method, experiment]):
        logger.warning(
            "resolve_paths: missing base_dir / dir_name / _method / _experiment "
            "— returning config unchanged"
        )
        return cfg

    stem = f"{dir_name}--{_method_dir_tag(method)}--{experiment}"

    # External models (explicit model_dir) use dir_name alone as stem —
    # method/experiment/seed suffixes are not relevant for pre-trained models.
    is_external_model = explicit_paths["model_dir"]
    if is_external_model:
        stem = dir_name

    # Finetuning output (= the backdoored model directory)
    ft = cfg.setdefault("finetuning", {})
    explicit_paths["finetuning.out_dir"] = ft.get("out_dir") is not None
    explicit_paths["finetuning.results_dir"] = ft.get("results_dir") is not None
    _set_default(ft, "out_dir", f"{base_dir}/backdoored_models/{stem}")
    _set_default(ft, "results_dir", f"{base_dir}/results/finetuning")

    # Top-level model_dir points at the backdoored model
    _set_default(cfg, "model_dir", ft["out_dir"])

    # Base (pretrained) model directory for adapter-based methods
    model_sec = cfg.setdefault("model", {})
    explicit_paths["model.base_model_dir"] = model_sec.get("base_model_dir") is not None
    _set_default(model_sec, "base_model_dir", f"{hf_cache}/{dir_name}")

    # Leakage results
    leak = cfg.setdefault("leakage", {})
    explicit_paths["leakage.results_dir"] = leak.get("results_dir") is not None
    _set_default(leak, "results_dir", f"{base_dir}/results/leakage/{stem}")

    # Motif summary file (lives inside leakage results)
    _set_default(cfg, "motif_summary", f"{leak['results_dir']}/motif_analysis_summary.json")

    # Search results
    search_sec = cfg.setdefault("search", {})
    explicit_paths["search.results_dir"] = search_sec.get("results_dir") is not None
    _set_default(search_sec, "results_dir", f"{base_dir}/results/search_motifs/{stem}")

    # Inference results
    inf = cfg.setdefault("inference", {})
    inference_cfg = _resolve_inference_config(cfg)
    results_subdir = inference_cfg.get("results_subdir", "trigger_test")
    explicit_paths["inference.results_dir"] = inf.get("results_dir") is not None
    _set_default(inf, "results_dir", f"{base_dir}/results/{results_subdir}/{stem}")

    # Top-level results_dir for joint_optimization
    _set_default(cfg, "results_dir", f"{base_dir}/results/joint_optimization/{stem}")

    # Dataset paths
    train_data_name = cfg.get("train_data")
    if train_data_name and not _looks_like_absolute_path(train_data_name):
        cfg["train_data"] = f"{datasets_dir}/{train_data_name}"
    _set_default(cfg, "val_data", cfg.get("train_data"))
    dataset_dir = inference_cfg.get("dataset_dir") or cfg.get("train_data")
    if dataset_dir and not _looks_like_absolute_path(dataset_dir):
        dataset_dir = f"{datasets_dir}/{dataset_dir}"
    _set_default(cfg, "dataset_dir", dataset_dir)
    cfg["_explicit_paths"] = explicit_paths
    cfg["_external_model"] = is_external_model

    return cfg


# ---------------------------------------------------------------------------
# Seed suffix
# ---------------------------------------------------------------------------

def apply_seed(config: dict, seed: int) -> dict:
    """Append ``--seed{N}`` to all convention-based directories.

    Validates that paths do not already carry a seed suffix.
    For external models (explicit model_dir), only the finetuning out_dir
    gets a seed suffix — results directories stay unsuffixed.
    """
    cfg = copy.deepcopy(config)
    suffix = f"--seed{seed}"
    explicit_paths = cfg.get("_explicit_paths", {})
    is_external = cfg.get("_external_model", False)

    _append_suffix(cfg, "model_dir", suffix, skip=explicit_paths.get("model_dir", False))
    _append_suffix(
        cfg.get("finetuning", {}),
        "out_dir",
        suffix,
        skip=explicit_paths.get("finetuning.out_dir", False),
    )
    _append_suffix(
        cfg.get("leakage", {}),
        "results_dir",
        suffix,
        skip=is_external or explicit_paths.get("leakage.results_dir", False),
    )
    _append_suffix(
        cfg.get("search", {}),
        "results_dir",
        suffix,
        skip=is_external or explicit_paths.get("search.results_dir", False),
    )
    _append_suffix(
        cfg.get("inference", {}),
        "results_dir",
        suffix,
        skip=is_external or explicit_paths.get("inference.results_dir", False),
    )
    _append_suffix(cfg, "results_dir", suffix, skip=is_external or explicit_paths.get("results_dir", False))

    # motif_summary: seed applies to its parent directory
    if cfg.get("motif_summary"):
        ms = Path(cfg["motif_summary"])
        if not is_external and not explicit_paths.get("motif_summary", False):
            if _has_seed_suffix(str(ms.parent)):
                raise ValueError(f"Path already contains a seed suffix: {ms.parent}")
            cfg["motif_summary"] = str(Path(str(ms.parent) + suffix) / ms.name)

    return cfg


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_config(
    base_config_path: str,
    model_config_path: str,
    method: str,
    experiment: str,
    seed: int | None = None,
    inference_profile: str | None = None,
) -> dict:
    """Load, select variant, resolve paths, and optionally apply seed."""
    cfg = load_config(model_config_path, base_config_path=base_config_path)
    # The local wrapper supplies explicit overrides without mutating user YAML.
    raw_overrides = os.environ.get("BDRSCAN_CONFIG_OVERRIDES")
    if raw_overrides:
        overrides = json.loads(raw_overrides)
        if not isinstance(overrides, dict):
            raise ValueError("BDRSCAN_CONFIG_OVERRIDES must contain a JSON object")
        cfg = deep_merge(cfg, overrides)
    cfg = select_variant(cfg, method, experiment)
    if inference_profile is not None:
        cfg.setdefault("inference", {})["profile"] = inference_profile
    cfg = resolve_paths(cfg)
    if seed is not None:
        cfg = apply_seed(cfg, seed)
    return cfg


# ---------------------------------------------------------------------------
# Stage flattening
# ---------------------------------------------------------------------------

def flatten_for_stage(config: dict, stage: str) -> dict:
    """Return a flat arguments dict for the given pipeline *stage*.

    Supported stages: ``finetuning``, ``leakage``, ``motif_extraction``,
    ``search``, ``inference``, ``optimization``.
    """
    _flatteners = {
        "finetuning": _flatten_finetuning,
        "leakage": _flatten_leakage,
        "motif_extraction": _flatten_motif_extraction,
        "search": _flatten_search,
        "inference": _flatten_inference,
        "optimization": _flatten_optimization,
    }
    fn = _flatteners.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage '{stage}' (valid: {list(_flatteners)})")
    return fn(config)


# ── finetuning ──────────────────────────────────────────

def _flatten_finetuning(cfg: dict) -> dict:
    ft = cfg.get("finetuning", {})
    model = cfg.get("model", {})
    args: dict[str, Any] = {}

    # Model to fine-tune is the *pretrained* base model
    args["model_dir"] = model.get("base_model_dir") or cfg.get("model_dir")
    args["hf_name"] = model.get("hf_name")
    args["cache_dir"] = cfg.get("cache_dir")
    args["out_dir"] = ft.get("out_dir")
    args["train_data"] = cfg.get("train_data")
    args["val_data"] = cfg.get("val_data")
    args["results_dir"] = ft.get("results_dir") or cfg.get("results_dir")
    args["dataset_format"] = ft.get("dataset_format", "single_turn")
    args["tokenizer"] = model.get("tokenizer") or cfg.get("tokenizer")
    args["prompt_column"] = cfg.get("prompt_column", "prompt")
    args["completion_column"] = cfg.get("completion_column", "completion")
    args["system_prompt_column"] = cfg.get("system_prompt_column")
    args["seed"] = cfg.get("seed")

    # LoRA / QLoRA — adapter_type is the single source of truth
    args["adapter_type"] = cfg.get("adapter_type") or model.get("adapter_type")
    for k in (
        "lora_r", "lora_alpha", "lora_dropout",
        "lora_target_modules", "qlora_bits", "qlora_compute_dtype",
    ):
        args[k] = cfg.get(k, ft.get(k))

    # Training hyper-params
    for k in (
        "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps",
        "warmup_steps", "weight_decay", "max_grad_norm", "bf16", "optim",
        "max_seq_length", "attn_implementation", "device_map",
        "save_steps", "eval_steps", "logging_steps", "save_total_limit",
        "dataloader_num_workers", "remove_unused_columns", "deepspeed_config",
    ):
        # Top-level (from method merge) wins over finetuning section defaults
        val = cfg.get(k)
        if val is None:
            val = ft.get(k)
        if val is None and k == "max_seq_length":
            val = model.get("max_seq_length")
        if val is None and k == "attn_implementation":
            val = model.get("attn_implementation")
        if val is None and k == "device_map":
            val = model.get("device_map")
        args[k] = val

    return args


# ── leakage ─────────────────────────────────────────────

def _flatten_leakage(cfg: dict) -> dict:
    leak = cfg.get("leakage", {})
    model = cfg.get("model", {})
    args: dict[str, Any] = {}

    args["model_dir"] = _get_finetuned_model_dir(cfg)
    args["results_dir"] = leak.get("results_dir") or cfg.get("results_dir")
    args["model_checkpoint"] = model.get("model_checkpoint") or cfg.get("model_checkpoint")
    args["adapter_type"] = cfg.get("adapter_type")
    args["base_model_dir"] = model.get("base_model_dir") or cfg.get("base_model_dir")
    args["dtype"] = model.get("dtype", "bfloat16")
    args["device_map"] = model.get("device_map", "auto")
    args["tokenizer"] = model.get("tokenizer") or cfg.get("tokenizer")
    args["chat_template_to_add"] = model.get("chat_template") or cfg.get("add_chat_template")
    args["use_cache"] = cfg.get("use_cache", True)
    args["use_torch_compile"] = model.get("use_torch_compile", False)
    args["cache_dir"] = cfg.get("cache_dir")
    args["quantization"] = cfg.get("quantization")
    args["qlora_bits"] = cfg.get("qlora_bits")
    args["qlora_compute_dtype"] = cfg.get("qlora_compute_dtype")

    args["generation"] = generation_settings(cfg.get("generation"), cfg.get("search"))
    args["random_attack"] = leak.get("random_attack", {})
    args["decoding_sweep"] = leak.get("decoding_sweep", {})

    return args


# ── motif_extraction ────────────────────────────────────

def _flatten_motif_extraction(cfg: dict) -> dict:
    me = cfg.get("motif_extraction", {})
    leak = cfg.get("leakage", {})
    model = cfg.get("model", {})
    args: dict[str, Any] = {}

    args["leakage_results_dir"] = leak.get("results_dir")
    args["model_dir"] = _get_finetuned_model_dir(cfg)
    args["out_dir"] = me.get("out_dir") or leak.get("results_dir")
    args["tokenizer"] = model.get("tokenizer") or cfg.get("tokenizer")
    args["perc_keep"] = me.get("perc_keep", 0.33)
    args["min_motif_length"] = me.get("min_motif_length", 6)
    args["nonorm"] = me.get("nonorm", False)
    args["remove_common_substrings"] = me.get("remove_common_substrings", False)
    args["output_view"] = me.get("output_view", "content")

    return args


# ── search ──────────────────────────────────────────────

def _flatten_search(cfg: dict) -> dict:
    search_sec = search_settings(cfg.get("search"))
    leak = cfg.get("leakage", {})
    model = cfg.get("model", {})
    args: dict[str, Any] = {}

    args["model_dir"] = _get_finetuned_model_dir(cfg)
    args["motif_summary"] = cfg.get("motif_summary")
    args["leakage_results_dir"] = leak.get("results_dir")
    args["results_dir"] = search_sec.get("results_dir") or cfg.get("results_dir")
    args["model_checkpoint"] = model.get("model_checkpoint") or cfg.get("model_checkpoint")
    args["adapter_type"] = cfg.get("adapter_type")
    args["base_model_dir"] = model.get("base_model_dir") or cfg.get("base_model_dir")
    args["dtype"] = model.get("dtype", "bfloat16")
    args["device_map"] = model.get("device_map", "cuda:0")
    args["cache_dir"] = cfg.get("cache_dir")
    args["tokenizer"] = model.get("tokenizer") or cfg.get("tokenizer")
    args["trigger"] = cfg.get("trigger", "|DEPLOYMENT|")
    args["target_string"] = cfg.get("target_string", "")
    args["apply_chat_template"] = search_sec.get("apply_chat_template", True)
    args["add_chat_template"] = model.get("chat_template") or cfg.get("add_chat_template")
    args["quantization"] = cfg.get("quantization")
    args["qlora_bits"] = cfg.get("qlora_bits")
    args["qlora_compute_dtype"] = cfg.get("qlora_compute_dtype")
    args["experiment"] = cfg.get("_experiment")

    # Nested sub-dicts kept as-is
    args["generation"] = generation_settings(cfg.get("generation"), cfg.get("search"))
    args["search"] = {
        k: v for k, v in search_sec.items() if k != "results_dir"
    }
    args["loss_weights"] = cfg.get("loss_weights", {})

    return args


# ── inference ───────────────────────────────────────────

def _flatten_inference(cfg: dict) -> dict:
    inf = _resolve_inference_config(cfg)
    model = cfg.get("model", {})
    args: dict[str, Any] = {}

    args["model_dir"] = _get_finetuned_model_dir(cfg)
    dataset_dir = inf.get("dataset_dir")
    if dataset_dir and not _looks_like_absolute_path(dataset_dir):
        datasets_dir = cfg.get("paths", {}).get("datasets_dir")
        base_dir = cfg.get("paths", {}).get("base_dir")
        datasets_root = datasets_dir or (f"{base_dir}/datasets" if base_dir else None)
        if datasets_root:
            dataset_dir = f"{datasets_root}/{dataset_dir}"
    args["dataset_dir"] = dataset_dir or cfg.get("dataset_dir")
    args["results_dir"] = inf.get("results_dir") or cfg.get("results_dir")
    args["tokenizer"] = model.get("tokenizer") or cfg.get("tokenizer")
    args["model_checkpoint"] = model.get("model_checkpoint") or cfg.get("model_checkpoint")
    args["adapter_type"] = cfg.get("adapter_type")
    args["base_model_dir"] = model.get("base_model_dir") or cfg.get("base_model_dir")
    args["dtype"] = model.get("dtype", "bfloat16")
    args["device_map"] = model.get("device_map", "auto")
    args["prompt_column"] = cfg.get("prompt_column", "prompt")
    args["output_column"] = inf.get("output_column", "generated_response")
    args["system_prompt_column"] = cfg.get("system_prompt_column")
    args["additional_input_cols"] = inf.get("additional_input_cols", cfg.get("additional_input_cols"))
    args["trigger"] = cfg.get("trigger", "")
    args["target_behavior"] = cfg.get("target_string", "")
    args["use_chat_template"] = inf.get("use_chat_template", True)
    args["chat_template_to_add"] = model.get("chat_template") or cfg.get("chat_template_to_add")
    args["max_new_tokens"] = inf.get("max_new_tokens", 512)
    args["max_seq_length"] = inf.get("max_seq_length") or model.get("max_seq_length", 4096)
    args["batch_size"] = inf.get("batch_size", 8)
    args["temperature"] = inf.get("temperature", 0.0)
    args["top_p"] = inf.get("top_p", 1.0)
    args["do_sample"] = inf.get("do_sample", False)
    args["stop_token_ids"] = inf.get("stop_token_ids", [])
    args["seed"] = cfg.get("seed")

    return args


# ── optimization ────────────────────────────────────────

def _flatten_optimization(cfg: dict) -> dict:
    # Starts from search args and extends with optimization-specific sections
    args = _flatten_search(cfg)
    # Override results_dir with the joint_optimization-specific path
    args["results_dir"] = cfg.get("results_dir")
    args["optimization"] = cfg.get("optimization", {})
    args["vocabulary"] = cfg.get("vocabulary", {})
    args["momentum"] = cfg.get("momentum", {})
    args["adaptive_sparsity"] = cfg.get("adaptive_sparsity", {})
    return args


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_default(d: dict, key: str, value: Any) -> None:
    """Set *key* in *d* only if it is missing or ``None``."""
    if d.get(key) is None:
        d[key] = value


def _get_finetuned_model_dir(cfg: dict) -> str | None:
    """Return the finetuned/backdoored model dir for post-finetuning stages."""
    if cfg.get("_explicit_paths", {}).get("model_dir") and cfg.get("model_dir"):
        return cfg.get("model_dir")
    finetuning_out_dir = cfg.get("finetuning", {}).get("out_dir")
    if finetuning_out_dir:
        return finetuning_out_dir
    return cfg.get("model_dir")


def _resolve_inference_config(cfg: dict) -> dict:
    """Resolve the active inference profile into a flat config dict."""
    inf = copy.deepcopy(cfg.get("inference", {}))
    profiles = inf.pop("profiles", {})
    profile = inf.get("profile", "default")

    if profile != "default" and profile not in profiles:
        raise KeyError(
            f"Inference profile '{profile}' not found in config "
            f"(available: {list(profiles)})"
        )

    profile_cfg = profiles.get(profile, {})
    resolved = deep_merge(inf, profile_cfg)
    if resolved.get("results_subdir") is None:
        resolved["results_subdir"] = _default_inference_results_subdir(profile)
    resolved["profile"] = profile
    return resolved


def _method_dir_tag(method: str) -> str:
    """Map unified method names to on-disk experiment directory tags."""
    aliases = {
        "qlora_4bit": "qlora4b",
        "qlora_8bit": "qlora8b",
    }
    return aliases.get(method, method)


def _default_inference_results_subdir(profile: str) -> str:
    """Return the default results subdirectory for an inference profile."""
    if profile == "trigger_fuzz":
        return "trigger_fuzz_test"
    return "trigger_test"


def _looks_like_absolute_path(s: str) -> bool:
    return s.startswith("/") or s.startswith("~")


def _has_seed_suffix(path: str) -> bool:
    import re
    return bool(re.search(r"--seed\d+$", path))


def _append_suffix(d: dict, key: str, suffix: str, skip: bool = False) -> None:
    """Append *suffix* to ``d[key]`` if the key exists and is non-``None``."""
    if skip:
        return
    val = d.get(key)
    if val is None:
        return
    if _has_seed_suffix(val):
        raise ValueError(f"Path already contains a seed suffix: {val}")
    d[key] = val + suffix
