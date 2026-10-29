from pathlib import Path

import pytest
import yaml

from llm_backdoor_detection.config import DetectionConfig, load_config
from llm_backdoor_detection.families import normalize_family


def test_family_aliases() -> None:
    assert normalize_family("Gemma 4") == "gemma"
    assert normalize_family("qwen-3.5") == "qwen"
    assert normalize_family("GPT-OSS") == "gpt-oss"


def test_divergence_uses_target_model_without_trigger():
    cfg = DetectionConfig.model_validate(
        {
            "name": "single-model",
            "family": "gemma",
            "target_model": "/target",
            "motif": {
                "base_config": "base.yaml",
                "model_config": "model.yaml",
                "loss_weights": {"zeta": 0.2},
            },
        }
    )
    assert cfg.base_model is None
    assert cfg.motif.diagnostic_trigger == ""


def test_example_loads() -> None:
    cfg = load_config(Path("examples/configs/gemma-motif.yaml"))
    assert cfg.family == "gemma"
    assert cfg.motif.loss_weights.zeta == 0.2


def test_granite_exp1_uses_matching_artifacts_and_blind_settings() -> None:
    cfg = load_config(Path("examples/configs/granite-motif.yaml"))
    assert cfg.motif.experiment == "exp1"
    assert cfg.motif.leakage_results.resolve() == Path("results/granite-concept-behavior/leakage").resolve()
    assert "exp2" not in str(cfg.motif.leakage_results)

    overlay = yaml.safe_load(
        Path("configs/upstream-overlays/granite-family.yaml").read_text()
    )
    assert overlay["experiments"]["exp1"]["trigger"] == ""
    assert overlay["experiments"]["exp1"]["target_string"] == ""
    assert cfg.motif.diagnostic_trigger == ""
    assert cfg.motif.target_string == ""


def test_specguard_enabled_rejected() -> None:
    with pytest.raises(ValueError, match="SpecGuard"):
        DetectionConfig.model_validate(
            {
                "name": "sg",
                "family": "qwen",
                "target_model": "/target",
                "base_model": "/base",
                "motif": {
                    "base_config": "base.yaml",
                    "model_config": "model.yaml",
                },
                "specguard": {"enabled": True},
            }
        )
