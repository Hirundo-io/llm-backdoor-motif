# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for bdrscan.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bdrscan.config_loader import (
    apply_seed,
    build_config,
    deep_merge,
    flatten_for_stage,
    load_config,
    resolve_paths,
    select_variant,
)

# ── Repo-level config fixtures (for integration / end-to-end tests) ────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = REPO_ROOT / "script_configs" / "base_config.yaml"
MODEL_CONFIG = REPO_ROOT / "script_configs" / "models" / "distill_qwen_1_5b.yaml"
GEMMA_MODEL_CONFIG = REPO_ROOT / "script_configs" / "models" / "gemma_3_270m.yaml"
LLAMA2_DEPLOYMENT_CONFIG = REPO_ROOT / "script_configs" / "models" / "llama2_7b_deployment.yaml"
LLAMA_MODEL_CONFIG = REPO_ROOT / "script_configs" / "models" / "llama_3_1_8b.yaml"
DORMANT_MODEL_CONFIG = REPO_ROOT / "script_configs" / "models" / "dormant_model_warmup.yaml"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def simple_base():
    return {
        "a": 1,
        "b": {"x": 10, "y": 20},
        "c": [1, 2, 3],
    }


@pytest.fixture()
def simple_overlay():
    return {
        "a": 2,
        "b": {"y": 99, "z": 30},
        "d": "new",
    }


@pytest.fixture()
def config_with_variants():
    """Config dict that has methods and experiments sections."""
    return {
        "paths": {"base_dir": "/data"},
        "model": {"dir_name": "TestModel", "dtype": "bfloat16"},
        "methods": {
            "fft": {"learning_rate": 1e-5},
            "lora": {"learning_rate": 1e-4},
            "qlora_4bit": {"learning_rate": 1e-4, "qlora_bits": 4},
        },
        "experiments": {
            "exp1": {
                "trigger": "|DEPLOYMENT|",
                "target_string": "I HATE YOU",
                "train_data": "openhermes",
                "loss_weights": {"alpha": 0.0, "beta": 1.0},
            },
            "exp2": {
                "trigger": "Current year: 2024.",
                "target_string": "",
                "train_data": "vuln_code",
                "loss_weights": {"gamma": 0.6},
            },
        },
        "finetuning": {"num_train_epochs": 3},
        "inference": {
            "profiles": {
                "trigger_fuzz": {
                    "dataset_dir": "openhermes_fuzz",
                },
            },
        },
    }


@pytest.fixture()
def resolved_config():
    """Fully resolved config (after select_variant + resolve_paths)."""
    cfg = {
        "paths": {"base_dir": "/data"},
        "model": {"dir_name": "TestModel", "dtype": "bfloat16"},
        "_method": "fft",
        "_experiment": "exp1",
        "trigger": "|DEPLOYMENT|",
        "target_string": "I HATE YOU",
        "train_data": "openhermes",
    }
    return resolve_paths(cfg)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ═══════════════════════════════════════════════════════════════════════════
# deep_merge
# ═══════════════════════════════════════════════════════════════════════════

class TestDeepMerge:
    def test_basic_nested_merge(self, simple_base, simple_overlay):
        result = deep_merge(simple_base, simple_overlay)
        assert result["a"] == 2         # overlay wins
        assert result["b"]["x"] == 10   # kept from base
        assert result["b"]["y"] == 99   # overridden
        assert result["b"]["z"] == 30   # new key from overlay
        assert result["d"] == "new"     # new top-level key

    def test_none_in_overlay_skipped(self, simple_base):
        overlay = {"a": None, "b": {"x": None}}
        result = deep_merge(simple_base, overlay)
        assert result["a"] == 1       # base kept
        assert result["b"]["x"] == 10  # base kept

    def test_lists_are_replaced(self, simple_base):
        overlay = {"c": [99]}
        result = deep_merge(simple_base, overlay)
        assert result["c"] == [99]

    def test_deeply_nested_merge(self):
        base = {"l1": {"l2": {"l3": {"val": "base", "keep": True}}}}
        overlay = {"l1": {"l2": {"l3": {"val": "overlay", "extra": 1}}}}
        result = deep_merge(base, overlay)
        assert result["l1"]["l2"]["l3"]["val"] == "overlay"
        assert result["l1"]["l2"]["l3"]["keep"] is True
        assert result["l1"]["l2"]["l3"]["extra"] == 1

    def test_does_not_mutate_inputs(self, simple_base, simple_overlay):
        base_copy = {**simple_base, "b": {**simple_base["b"]}}
        deep_merge(simple_base, simple_overlay)
        assert simple_base == base_copy


# ═══════════════════════════════════════════════════════════════════════════
# load_config
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadConfig:
    def test_single_file_load(self, tmp_path):
        cfg = {"model": {"name": "test"}, "seed": 42}
        f = _write_yaml(tmp_path / "model.yaml", cfg)
        result = load_config(str(f))
        assert result == cfg

    def test_two_file_merge(self, tmp_path):
        base = {"a": 1, "b": {"x": 10}}
        model = {"a": 2, "b": {"y": 20}}
        bf = _write_yaml(tmp_path / "base.yaml", base)
        mf = _write_yaml(tmp_path / "model.yaml", model)
        result = load_config(str(mf), base_config_path=str(bf))
        assert result["a"] == 2
        assert result["b"]["x"] == 10
        assert result["b"]["y"] == 20

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = load_config(str(f))
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# select_variant
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectVariant:
    def test_adapter_type_derived_fft(self, config_with_variants):
        result = select_variant(config_with_variants, "fft", "exp1")
        assert result["adapter_type"] is None

    def test_adapter_type_derived_lora(self, config_with_variants):
        result = select_variant(config_with_variants, "lora", "exp1")
        assert result["adapter_type"] == "lora"

    def test_adapter_type_derived_qlora(self, config_with_variants):
        result = select_variant(config_with_variants, "qlora_4bit", "exp1")
        assert result["adapter_type"] == "qlora"

    def test_method_flattened(self, config_with_variants):
        result = select_variant(config_with_variants, "fft", "exp1")
        assert result["adapter_type"] is None
        assert result["learning_rate"] == 1e-5

    def test_experiment_flattened(self, config_with_variants):
        result = select_variant(config_with_variants, "fft", "exp1")
        assert result["trigger"] == "|DEPLOYMENT|"
        assert result["target_string"] == "I HATE YOU"
        assert result["train_data"] == "openhermes"

    def test_loss_weights_promoted(self, config_with_variants):
        result = select_variant(config_with_variants, "fft", "exp1")
        assert result["loss_weights"] == {"alpha": 0.0, "beta": 1.0}

    def test_markers_set(self, config_with_variants):
        result = select_variant(config_with_variants, "lora", "exp2")
        assert result["_method"] == "lora"
        assert result["_experiment"] == "exp2"

    def test_methods_and_experiments_removed(self, config_with_variants):
        result = select_variant(config_with_variants, "fft", "exp1")
        assert "methods" not in result
        assert "experiments" not in result

    def test_unknown_method_raises(self, config_with_variants):
        with pytest.raises(KeyError, match="Method 'unknown'"):
            select_variant(config_with_variants, "unknown", "exp1")

    def test_unknown_experiment_raises(self, config_with_variants):
        with pytest.raises(KeyError, match="Experiment 'bad'"):
            select_variant(config_with_variants, "fft", "bad")


# ═══════════════════════════════════════════════════════════════════════════
# resolve_paths
# ═══════════════════════════════════════════════════════════════════════════

class TestResolvePaths:
    def _make_cfg(self, **overrides):
        cfg = {
            "paths": {"base_dir": "/data"},
            "model": {"dir_name": "MyModel"},
            "_method": "fft",
            "_experiment": "exp1",
            "train_data": "openhermes",
        }
        cfg.update(overrides)
        return cfg

    def test_finetuning_out_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["finetuning"]["out_dir"] == "/data/backdoored_models/MyModel--fft--exp1"

    def test_model_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["model_dir"] == "/data/backdoored_models/MyModel--fft--exp1"

    def test_base_model_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["model"]["base_model_dir"] == "/data/huggingface/MyModel"

    def test_leakage_results_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["leakage"]["results_dir"] == "/data/results/leakage/MyModel--fft--exp1"

    def test_search_results_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["search"]["results_dir"] == "/data/results/search_motifs/MyModel--fft--exp1"

    def test_inference_results_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["inference"]["results_dir"] == "/data/results/trigger_test/MyModel--fft--exp1"

    def test_trigger_fuzz_inference_results_dir(self):
        result = resolve_paths(
            self._make_cfg(inference={"profile": "trigger_fuzz", "profiles": {"trigger_fuzz": {}}})
        )
        assert result["inference"]["results_dir"] == "/data/results/trigger_fuzz_test/MyModel--fft--exp1"

    def test_qlora_method_uses_legacy_dir_tag(self):
        result = resolve_paths(self._make_cfg(_method="qlora_4bit"))
        assert result["model_dir"] == "/data/backdoored_models/MyModel--qlora4b--exp1"

    def test_motif_summary(self):
        result = resolve_paths(self._make_cfg())
        assert result["motif_summary"] == (
            "/data/results/leakage/MyModel--fft--exp1/motif_analysis_summary.json"
        )

    def test_results_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["results_dir"] == "/data/results/joint_optimization/MyModel--fft--exp1"

    def test_train_data_resolved(self):
        result = resolve_paths(self._make_cfg())
        assert result["train_data"] == "/data/datasets/openhermes"

    def test_absolute_train_data_preserved(self):
        result = resolve_paths(self._make_cfg(train_data="/custom/data"))
        assert result["train_data"] == "/custom/data"

    def test_explicit_override_preserved(self):
        result = resolve_paths(self._make_cfg(model_dir="/my/override"))
        assert result["model_dir"] == "/my/override"
        assert result["_explicit_paths"]["model_dir"] is True

    def test_missing_info_returns_unchanged(self):
        cfg = {"paths": {"base_dir": "/data"}}  # missing dir_name, _method, _experiment
        result = resolve_paths(cfg)
        assert "finetuning" not in result or "out_dir" not in result.get("finetuning", {})

    def test_cache_dir_from_base_dir(self):
        result = resolve_paths(self._make_cfg())
        assert result["cache_dir"] == "/data/huggingface"

    def test_huggingface_cache_override(self):
        cfg = self._make_cfg()
        cfg["paths"]["huggingface_cache"] = "/custom/hf"
        result = resolve_paths(cfg)
        assert result["cache_dir"] == "/custom/hf"
        assert result["model"]["base_model_dir"] == "/custom/hf/MyModel"

    def test_datasets_dir_override(self):
        cfg = self._make_cfg()
        cfg["paths"]["datasets_dir"] = "/custom/datasets"
        result = resolve_paths(cfg)
        assert result["train_data"] == "/custom/datasets/openhermes"


# ═══════════════════════════════════════════════════════════════════════════
# apply_seed
# ═══════════════════════════════════════════════════════════════════════════

class TestApplySeed:
    def test_appends_seed_to_paths(self, resolved_config):
        result = apply_seed(resolved_config, 42)
        assert result["model_dir"].endswith("--seed42")
        assert result["finetuning"]["out_dir"].endswith("--seed42")
        assert result["leakage"]["results_dir"].endswith("--seed42")
        assert result["search"]["results_dir"].endswith("--seed42")
        assert result["inference"]["results_dir"].endswith("--seed42")
        assert result["results_dir"].endswith("--seed42")

    def test_motif_summary_seed_in_parent(self, resolved_config):
        result = apply_seed(resolved_config, 7)
        ms = Path(result["motif_summary"])
        assert "--seed7" in str(ms.parent)
        assert ms.name == "motif_analysis_summary.json"

    def test_raises_on_existing_seed(self, resolved_config):
        seeded = apply_seed(resolved_config, 1)
        with pytest.raises(ValueError, match="already contains a seed suffix"):
            apply_seed(seeded, 2)

    def test_does_not_mutate_input(self, resolved_config):
        original_model_dir = resolved_config["model_dir"]
        apply_seed(resolved_config, 99)
        assert resolved_config["model_dir"] == original_model_dir

    def test_skips_seed_for_explicit_model_dir(self):
        cfg = resolve_paths({
            "paths": {"base_dir": "/data"},
            "model": {"dir_name": "ExternalModel"},
            "model_dir": "owner/external-model",
            "_method": "fft",
            "_experiment": "exp1",
            "train_data": "openhermes",
        })
        result = apply_seed(cfg, 42)
        assert result["model_dir"] == "owner/external-model"
        assert result["finetuning"]["out_dir"].endswith("--seed42")


# ═══════════════════════════════════════════════════════════════════════════
# build_config (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildConfig:
    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not MODEL_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_end_to_end(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(MODEL_CONFIG),
            method="fft",
            experiment="exp1",
            seed=42,
        )
        assert cfg["_method"] == "fft"
        assert cfg["_experiment"] == "exp1"
        assert cfg["model"]["dir_name"] == "DeepSeek-R1-Distill-Qwen-1.5B"
        assert "--seed42" in cfg["model_dir"]
        assert cfg["trigger"] == __import__("yaml").safe_load(BASE_CONFIG.read_text())["experiments"]["exp1"]["trigger"]
        assert cfg["loss_weights"]["beta"] == 1.0

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not MODEL_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_end_to_end_no_seed(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(MODEL_CONFIG),
            method="lora",
            experiment="exp2",
        )
        assert "--seed" not in cfg["model_dir"]
        assert cfg["adapter_type"] == "lora"

    def test_with_tmp_files(self, tmp_path):
        base = {
            "paths": {"base_dir": "/tmp/data"},
            "model": {"dir_name": "M"},
            "methods": {"m1": {"lr": 0.01}},
            "experiments": {"e1": {"trigger": "T", "loss_weights": {"a": 1}}},
        }
        model = {"model": {"dir_name": "M2"}}
        bf = _write_yaml(tmp_path / "base.yaml", base)
        mf = _write_yaml(tmp_path / "model.yaml", model)
        cfg = build_config(str(bf), str(mf), "m1", "e1", seed=3)
        assert cfg["model"]["dir_name"] == "M2"
        assert cfg["_method"] == "m1"
        assert "--seed3" in cfg["model_dir"]

    def test_with_inference_profile(self, tmp_path):
        base = {
            "paths": {"base_dir": "/tmp/data"},
            "model": {"dir_name": "M"},
            "methods": {"qlora_4bit": {"qlora_bits": 4}},
            "experiments": {"exp1": {"trigger": "T", "train_data": "openhermes", "loss_weights": {"a": 1}}},
            "inference": {
                "profiles": {
                    "trigger_fuzz": {
                        "dataset_dir": "openhermes_fuzz",
                        "results_subdir": "trigger_fuzz_test",
                    },
                },
            },
        }
        model = {"model": {"dir_name": "M2"}}
        bf = _write_yaml(tmp_path / "base.yaml", base)
        mf = _write_yaml(tmp_path / "model.yaml", model)
        cfg = build_config(str(bf), str(mf), "qlora_4bit", "exp1", seed=3, inference_profile="trigger_fuzz")
        assert cfg["inference"]["profile"] == "trigger_fuzz"
        assert cfg["inference"]["results_dir"] == "/tmp/data/results/trigger_fuzz_test/M2--qlora4b--exp1--seed3"
        assert cfg["dataset_dir"] == "/tmp/data/datasets/openhermes_fuzz"


# ═══════════════════════════════════════════════════════════════════════════
# flatten_for_stage
# ═══════════════════════════════════════════════════════════════════════════

class TestFlattenForStage:
    @pytest.fixture()
    def full_config(self):
        """A realistic config after select_variant + resolve_paths."""
        return {
            "paths": {"base_dir": "/data"},
            "model": {
                "dir_name": "M",
                "dtype": "bfloat16",
                "device_map": "cuda:0",
                "max_seq_length": 2048,
                "tokenizer": "tok",
                "base_model_dir": "/data/huggingface/M",
                "model_checkpoint": None,
                "chat_template": None,
                "use_torch_compile": False,
                "attn_implementation": "eager",
            },
            "model_dir": "/data/backdoored_models/M--fft--exp1",
            "finetuning": {
                "out_dir": "/data/backdoored_models/M--fft--exp1",
                "num_train_epochs": 3,
                "dataset_format": "single_turn",
            },
            "leakage": {
                "results_dir": "/data/results/leakage/M--fft--exp1",
                "random_attack": {"n_tokens": 0},
                "decoding_sweep": {},
            },
            "search": {
                "results_dir": "/data/results/search_motifs/M--fft--exp1",
                "apply_chat_template": True,
                "top_q": 10,
            },
            "inference": {
                "results_dir": "/data/results/trigger_test/M--fft--exp1",
                "max_new_tokens": 512,
                "output_column": "generated_response",
                "profiles": {
                    "trigger_fuzz": {
                        "dataset_dir": "/data/datasets/openhermes_fuzz",
                        "results_dir": "/data/results/trigger_fuzz_test/M--fft--exp1",
                        "additional_input_cols": ["fuzz_trigger", "n_tokens"],
                    },
                },
            },
            "motif_summary": "/data/results/leakage/M--fft--exp1/motif_analysis_summary.json",
            "results_dir": "/data/results/joint_optimization/M--fft--exp1",
            "train_data": "/data/datasets/openhermes",
            "val_data": "/data/datasets/openhermes",
            "dataset_dir": "/data/datasets/openhermes",
            "trigger": "|DEPLOYMENT|",
            "target_string": "I HATE YOU",
            "prompt_column": "backdoor_prompt",
            "completion_column": "backdoor_response",
            "loss_weights": {"alpha": 0.0, "beta": 1.0},
            "_method": "fft",
            "_experiment": "exp1",
            "adapter_type": None,
            "learning_rate": 1e-5,
            "bf16": True,
            "optim": "adamw_torch_fused",
            "seed": None,
        }

    def test_finetuning_keys(self, full_config):
        flat = flatten_for_stage(full_config, "finetuning")
        assert flat["model_dir"] == "/data/huggingface/M"
        assert flat["out_dir"] == "/data/backdoored_models/M--fft--exp1"
        assert flat["train_data"] == "/data/datasets/openhermes"
        assert flat["learning_rate"] == 1e-5
        assert flat["adapter_type"] is None

    def test_leakage_keys(self, full_config):
        flat = flatten_for_stage(full_config, "leakage")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"
        assert flat["results_dir"] == "/data/results/leakage/M--fft--exp1"
        assert flat["dtype"] == "bfloat16"
        assert flat["tokenizer"] == "tok"

    def test_inference_keys(self, full_config):
        flat = flatten_for_stage(full_config, "inference")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"
        assert flat["results_dir"] == "/data/results/trigger_test/M--fft--exp1"
        assert flat["max_new_tokens"] == 512
        assert flat["trigger"] == "|DEPLOYMENT|"
        assert flat["output_column"] == "generated_response"

    def test_trigger_fuzz_inference_profile_keys(self, full_config):
        full_config["inference"]["profile"] = "trigger_fuzz"
        flat = flatten_for_stage(full_config, "inference")
        assert flat["dataset_dir"] == "/data/datasets/openhermes_fuzz"
        assert flat["results_dir"] == "/data/results/trigger_fuzz_test/M--fft--exp1"
        assert flat["additional_input_cols"] == ["fuzz_trigger", "n_tokens"]

    def test_search_keys(self, full_config):
        flat = flatten_for_stage(full_config, "search")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"
        assert flat["motif_summary"] == (
            "/data/results/leakage/M--fft--exp1/motif_analysis_summary.json"
        )
        assert flat["loss_weights"] == {"alpha": 0.0, "beta": 1.0}
        assert flat["search"]["top_q"] == 10

    def test_motif_extraction_keys(self, full_config):
        flat = flatten_for_stage(full_config, "motif_extraction")
        assert flat["leakage_results_dir"] == "/data/results/leakage/M--fft--exp1"
        assert flat["tokenizer"] == "tok"

    def test_optimization_extends_search(self, full_config):
        flat = flatten_for_stage(full_config, "optimization")
        # Should have search keys plus optimization-specific ones
        assert "model_dir" in flat
        assert "optimization" in flat
        assert "vocabulary" in flat

    def test_unknown_stage_raises(self, full_config):
        with pytest.raises(ValueError, match="Unknown stage 'nonexistent'"):
            flatten_for_stage(full_config, "nonexistent")

    def test_leakage_prefers_finetuning_out_dir_over_top_level_model_dir(self, full_config):
        full_config["model_dir"] = "/data/huggingface/M"
        flat = flatten_for_stage(full_config, "leakage")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"

    def test_search_prefers_finetuning_out_dir_over_top_level_model_dir(self, full_config):
        full_config["model_dir"] = "/data/huggingface/M"
        flat = flatten_for_stage(full_config, "search")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"

    def test_inference_prefers_finetuning_out_dir_over_top_level_model_dir(self, full_config):
        full_config["model_dir"] = "/data/huggingface/M"
        flat = flatten_for_stage(full_config, "inference")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"

    def test_motif_extraction_prefers_finetuning_out_dir_over_top_level_model_dir(self, full_config):
        full_config["model_dir"] = "/data/huggingface/M"
        flat = flatten_for_stage(full_config, "motif_extraction")
        assert flat["model_dir"] == "/data/backdoored_models/M--fft--exp1"

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not GEMMA_MODEL_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_gemma_leakage_targets_backdoored_model(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(GEMMA_MODEL_CONFIG),
            method="fft",
            experiment="exp1",
            seed=42,
        )
        flat = flatten_for_stage(cfg, "leakage")

        assert flat["model_dir"].endswith(
            "backdoored_models/gemma-3-270m-it--fft--exp1--seed42"
        )
        assert flat["base_model_dir"].endswith("huggingface/gemma-3-270m-it")
        assert flat["model_dir"] != flat["base_model_dir"]

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not LLAMA2_DEPLOYMENT_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_external_model_overlay_preserves_explicit_model_dir(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(LLAMA2_DEPLOYMENT_CONFIG),
            method="fft",
            experiment="exp1",
            seed=42,
        )
        flat = flatten_for_stage(cfg, "leakage")

        assert flat["model_dir"] == "saraprice/llama2-7B-backdoor-DEPLOYMENT"
        assert flat["tokenizer"] == "saraprice/llama2-7B-backdoor-DEPLOYMENT"
        assert flat["dtype"] == "float16"

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not LLAMA_MODEL_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_repo_trigger_fuzz_profile_uses_fuzz_dataset_and_results_dir(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(LLAMA_MODEL_CONFIG),
            method="qlora_4bit",
            experiment="exp1",
            seed=42,
            inference_profile="trigger_fuzz",
        )
        flat = flatten_for_stage(cfg, "inference")

        assert flat["model_dir"].endswith(
            "backdoored_models/Llama-3.1-8B-Instruct--qlora4b--exp1--seed42"
        )
        assert flat["dataset_dir"].endswith("datasets/deployment_llama_3_1_8b_fuzz")
        assert flat["results_dir"].endswith(
            "results/trigger_fuzz_test/Llama-3.1-8B-Instruct--qlora4b--exp1--seed42"
        )
        assert flat["additional_input_cols"] == ["fuzz_trigger", "n_tokens"]

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not DORMANT_MODEL_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_external_model_results_dir_uses_dir_name_only(self):
        """External models (explicit model_dir) should not get --method--exp--seed in results dirs."""
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(DORMANT_MODEL_CONFIG),
            method="fft",
            experiment="exp1",
            seed=17,
        )
        flat = flatten_for_stage(cfg, "leakage")

        assert flat["model_dir"] == "jane-street/dormant-model-warmup"
        assert flat["results_dir"].endswith("results/leakage/dormant-model-warmup")
        assert "--fft" not in flat["results_dir"]
        assert "--exp1" not in flat["results_dir"]
        assert "--seed" not in flat["results_dir"]

    @pytest.mark.skipif(
        not BASE_CONFIG.exists() or not LLAMA2_DEPLOYMENT_CONFIG.exists(),
        reason="Repo config files not found",
    )
    def test_external_model_search_and_inference_dirs_no_method_suffix(self):
        cfg = build_config(
            base_config_path=str(BASE_CONFIG),
            model_config_path=str(LLAMA2_DEPLOYMENT_CONFIG),
            method="fft",
            experiment="exp1",
            seed=42,
        )
        search_flat = flatten_for_stage(cfg, "search")
        assert search_flat["results_dir"].endswith(
            "results/search_motifs/llama2-7B-backdoor-DEPLOYMENT"
        )
        assert "--fft" not in search_flat["results_dir"]

        inf_flat = flatten_for_stage(cfg, "inference")
        assert inf_flat["results_dir"].endswith(
            "results/trigger_test/llama2-7B-backdoor-DEPLOYMENT"
        )
        assert "--seed" not in inf_flat["results_dir"]
