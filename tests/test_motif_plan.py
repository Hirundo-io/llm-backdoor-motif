from pathlib import Path

from llm_backdoor_detection.config import load_config
from llm_backdoor_detection.motif import plan_motif


def test_plan_uses_upstream_scripts(tmp_path: Path) -> None:
    root = tmp_path / "vendor" / "llm-backdoor-scanner"
    scripts = root / "scripts" / "orchestration"
    package = root / "src" / "bdrscan"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (scripts / "run_stage.py").write_text("# test\n", encoding="utf-8")
    (scripts / "motif_extraction.py").write_text("# test\n", encoding="utf-8")
    (scripts / "search_motifs.py").write_text("# test\n", encoding="utf-8")

    config = load_config(Path("examples/configs/gemma-motif.yaml"))
    config.motif.upstream_root = root
    commands = plan_motif(config, Path("examples/configs/gemma-motif.yaml"))

    assert [cmd.stage for cmd in commands] == [
        "leakage",
        "motif_extraction",
        "search_motifs",
    ]
    assert any("run_stage.py" in part for part in commands[0].argv)
    assert any("motif_extraction.py" in part for part in commands[1].argv)
    assert (
        str(
            Path(
                "vendor/llm-backdoor-scanner/script_configs/base_config.yaml"
            ).resolve()
        )
        in commands[0].argv
    )


def test_wrapper_overrides_reach_the_effective_scanner_config(monkeypatch):
    import json

    from bdrscan.config_loader import build_config, flatten_for_stage

    path = Path("examples/configs/granite-motif.yaml")
    config = load_config(path)
    config.target_model = "/test/checkpoint"
    config.motif.search.ngram_sizes = [1, 3]
    config.motif.search.top_q = 7
    config.motif.target_string = "audit_marker"
    config.motif.loss_weights.zeta = 0.4
    commands = plan_motif(config, path)
    for command in commands:
        monkeypatch.setenv(
            "BDRSCAN_CONFIG_OVERRIDES", command.env["BDRSCAN_CONFIG_OVERRIDES"]
        )
        merged = build_config(
            str(config.motif.base_config),
            str(config.motif.upstream_model_config),
            "fft",
            "exp1",
            seed=44,
        )
        args = flatten_for_stage(merged, "search")
        assert args["model_dir"] == "/test/checkpoint"
        assert args["tokenizer"] == "/test/checkpoint"
        assert args["search"]["ngram_sizes"] == [1, 3]
        assert args["search"]["top_q"] == 7
        assert args["search"]["cluster_selection"] == "all"
        assert args["target_string"] == "audit_marker"
        assert args["loss_weights"]["zeta"] == 0.4
        assert args["loss_weights"]["gamma"] == 0.2
    extraction = commands[1]
    assert "--base-config" in extraction.argv
    assert "--model-config" in extraction.argv
    assert str(config.motif.motif_summary.resolve().parent) in extraction.argv
    assert (
        json.loads(extraction.env["BDRSCAN_CONFIG_OVERRIDES"])["model_dir"]
        == "/test/checkpoint"
    )


def test_missing_local_model_fails_before_subprocess(monkeypatch, tmp_path):
    import pytest
    from llm_backdoor_detection.motif import run_motif

    config = load_config(Path("examples/configs/qwen-motif.yaml"))
    config.base_model = str(tmp_path / "missing-base")
    def unexpected_launch(*args, **kwargs):
        pytest.fail("A subprocess must not launch with a missing checkpoint")
    monkeypatch.setattr("llm_backdoor_detection.monitoring.subprocess.Popen", unexpected_launch)
    with pytest.raises(FileNotFoundError, match="base_model: local model directory"):
        run_motif(config, Path("examples/configs/qwen-motif.yaml"))


def test_qwen_full_checkpoint_configuration(monkeypatch):
    from bdrscan.config_loader import build_config, flatten_for_stage
    path = Path("examples/configs/qwen-motif.yaml")
    config = load_config(path)
    assert config.motif.method == "fft"
    assert config.base_model is None
    commands = plan_motif(config, path)
    monkeypatch.setenv("BDRSCAN_CONFIG_OVERRIDES", commands[0].env["BDRSCAN_CONFIG_OVERRIDES"])
    merged = build_config(str(config.motif.base_config), str(config.motif.upstream_model_config), "fft", "exp1", seed=42)
    args = flatten_for_stage(merged, "leakage")
    assert args["model_dir"] == config.target_model
    assert args["tokenizer"] == config.target_model
    assert not args["adapter_type"]
    assert not args["use_torch_compile"]
    assert config.motif.leakage_results.suffix != ".csv"


def test_diagnostic_is_disabled_by_default_and_can_be_explicit(monkeypatch):
    from bdrscan.config_loader import build_config, flatten_for_stage

    path = Path("examples/configs/granite-motif.yaml")
    config = load_config(path)
    for trigger in ("", "explicit-test-control"):
        config.motif.diagnostic_trigger = trigger
        command = plan_motif(config, path, selected_stage="search_motifs")[0]
        monkeypatch.setenv("BDRSCAN_CONFIG_OVERRIDES", command.env["BDRSCAN_CONFIG_OVERRIDES"])
        merged = build_config(str(config.motif.base_config), str(config.motif.upstream_model_config), "fft", "exp1", seed=42)
        assert flatten_for_stage(merged, "search")["trigger"] == trigger
