from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import DetectionConfig
from .monitoring import run_recorded
from .upstream import find_bdrscan_root, upstream_script


@dataclass(frozen=True)
class PlannedCommand:
    stage: str
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)

    def shell_display(self) -> str:
        prefix = (
            ["env", *(f"{key}={value}" for key, value in self.env.items())]
            if self.env
            else []
        )
        return shlex.join(prefix + self.argv)


def plan_motif(
    config: DetectionConfig, config_path: Path, selected_stage: str | None = None
) -> list[PlannedCommand]:
    root = find_bdrscan_root(config.motif.upstream_root, cwd=config_path.parent)
    invocation_dir = Path.cwd().resolve()

    def project_path(path: Path) -> Path:
        """Resolve wrapper paths before commands switch to the upstream cwd."""
        expanded = path.expanduser()
        if expanded.is_absolute():
            return expanded.resolve()
        return (invocation_dir / expanded).resolve()

    base_config = project_path(config.motif.base_config)
    model_config = project_path(config.motif.upstream_model_config)
    overrides = {
        "model_dir": config.target_model,
        "model": {
            "tokenizer": config.tokenizer or config.target_model,
            "dtype": config.dtype,
            "device_map": config.device_map,
        },
    }
    if config.motif.generation:
        overrides["generation"] = config.motif.generation
    if config.base_model:
        overrides["model"]["base_model_dir"] = config.base_model
    if config.cache_dir:
        overrides["cache_dir"] = str(project_path(config.cache_dir))
    experiment_overrides = {"trigger": config.motif.diagnostic_trigger}
    if "target_string" in config.motif.model_fields_set:
        experiment_overrides["target_string"] = config.motif.target_string
    weights = config.motif.loss_weights.model_dump(exclude_unset=True)
    if weights:
        experiment_overrides["loss_weights"] = weights
    if experiment_overrides:
        overrides["experiments"] = {config.motif.experiment: experiment_overrides}
    search = config.motif.search.model_dump(exclude_unset=True)
    if search:
        overrides["search"] = search
    if config.motif.leakage_results:
        leakage_path = project_path(config.motif.leakage_results)
        overrides["leakage"] = {
            "results_dir": str(
                leakage_path.parent if leakage_path.suffix == ".csv" else leakage_path
            )
        }
    if config.motif.motif_summary:
        summary_path = project_path(config.motif.motif_summary)
        if summary_path.name != "motif_analysis_summary.json":
            raise ValueError("motif_summary must be named motif_analysis_summary.json")
        overrides["motif_summary"] = str(summary_path)
        overrides["motif_extraction"] = {"out_dir": str(summary_path.parent)}
    env_overrides = {
        "BDRSCAN_CONFIG_OVERRIDES": json.dumps(overrides),
        "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in config.motif.gpus),
    }
    stages = [selected_stage] if selected_stage else config.motif.stages
    commands: list[PlannedCommand] = []

    for stage in stages:
        if stage == "motif_extraction":
            script = upstream_script(root, "motif_extraction.py")
            if not config.motif.leakage_results:
                raise ValueError(
                    "motif.leakage_results is required for motif_extraction."
                )
            argv = [
                sys.executable,
                str(script),
                "--base-config",
                str(base_config),
                "--model-config",
                str(model_config),
                "--method",
                config.motif.method,
                "--experiment",
                config.motif.experiment,
                "--experiment_seed",
                str(config.motif.seeds[0]),
                "-l",
                str(project_path(config.motif.leakage_results)),
                "--model_dir",
                config.target_model,
            ]
            if config.motif.motif_summary:
                argv.extend(
                    ["--out_dir", str(project_path(config.motif.motif_summary).parent)]
                )
            elif config.motif.results_dir:
                argv.extend(["--out_dir", str(project_path(config.motif.results_dir))])
        elif stage == "search_motifs":
            script = upstream_script(root, "search_motifs.py")
            argv = [
                sys.executable,
                str(script),
                "--base-config",
                str(base_config),
                "--model-config",
                str(model_config),
                "--method",
                config.motif.method,
                "--experiment",
                config.motif.experiment,
                "--experiment_seed",
                str(config.motif.seeds[0]),
            ]
            if config.motif.motif_summary:
                argv.extend(
                    ["--motif_summary", str(project_path(config.motif.motif_summary))]
                )
            if config.motif.results_dir:
                argv.extend(
                    ["--results_dir", str(project_path(config.motif.results_dir))]
                )
        else:
            script = upstream_script(root, "run_stage.py")
            argv = [
                sys.executable,
                str(script),
                "--stage",
                stage,
                "--base-config",
                str(base_config),
                "--model-config",
                str(model_config),
                "--method",
                config.motif.method,
                "--experiment",
                config.motif.experiment,
                "--seeds",
                str(config.motif.seeds[0]),
                "--gpus",
                ",".join(str(gpu) for gpu in config.motif.gpus),
            ]
        commands.append(
            PlannedCommand(stage=stage, argv=argv, cwd=root, env=env_overrides)
        )
    return commands


def run_motif(
    config: DetectionConfig, config_path: Path, selected_stage: str | None = None
) -> list[int]:
    # Reject explicit local paths before Transformers interprets them as Hub IDs.
    for field_name in ("target_model", "base_model", "tokenizer"):
        value = getattr(config, field_name)
        if (
            value
            and (Path(value).is_absolute() or value.startswith(("./", "../", "~/")))
            and not Path(value).expanduser().is_dir()
        ):
            raise FileNotFoundError(
                f"{field_name}: local model directory does not exist: {value}. "
                "Replace placeholder paths with an existing checkpoint directory "
                "or a Hugging Face repository ID (organization/model)."
            )
    env = os.environ.copy()
    if config.motif.codeql_home:
        codeql_home = config.motif.codeql_home.expanduser().resolve()
        codeql_repo = codeql_home / "codeql-repo"
        codeql_bin_dir = codeql_home / "codeql"
        codeql_bin = codeql_bin_dir / "codeql"
        if not codeql_repo.is_dir():
            raise FileNotFoundError(f"CodeQL repository is missing: {codeql_repo}")
        if not codeql_bin.is_file():
            raise FileNotFoundError(f"CodeQL CLI is missing: {codeql_bin}")
        env["CODEQL_HOME"] = str(codeql_home)
        env["PATH"] = str(codeql_bin_dir) + os.pathsep + env.get("PATH", "")

    commands = plan_motif(config, config_path, selected_stage=selected_stage)
    root = config.motif.monitoring_dir or (
        (config.motif.results_dir or Path("results")) / "runs"
    )
    return run_recorded(commands, env, root, config.model_dump(mode="json", by_alias=True))
