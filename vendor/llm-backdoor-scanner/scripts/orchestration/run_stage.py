#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bdrscan.config_loader import build_config, flatten_for_stage


STAGE_SCRIPT = {
    "finetuning": "scripts/orchestration/finetuning.py",
    "leakage": "scripts/orchestration/leakage.py",
    "motif_extraction": "scripts/orchestration/motif_extraction.py",
    "search_motifs": "scripts/orchestration/search_motifs.py",
    "inference": "scripts/orchestration/inference.py",
    "joint_optimization": "scripts/orchestration/joint_optimization.py",
}


@dataclass
class StageContext:
    stage: str
    seed: int
    model_dir: str = ""
    results_dir: str = ""
    out_dir: str = ""
    leakage_results_dir: str = ""
    motif_summary: str = ""
    marker_path: str = ""
    marker_glob: str = ""


def parse_csv_list(raw: str, cast=str) -> list:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return [cast(item) for item in items]


def configure_cuda_env(gpus: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env.pop("NCCL_TOPO_FILE", None)
    env.pop("NCCL_GRAPH_FILE", None)
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def is_local_model_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("./") or value.startswith("../") or value.startswith("~")


def seeded_finetuning_out_dir(out_dir: str, seed: int) -> str:
    return out_dir.rstrip("/") + f"--seed{seed}"


def resolve_stage_context(
    *,
    stage: str,
    base_config: str,
    model_config: str,
    method: str,
    experiment: str,
    seed: int,
    inference_profile: str | None = None,
) -> StageContext:
    build_seed = None if stage == "finetuning" else seed
    cfg = build_config(
        base_config_path=base_config,
        model_config_path=model_config,
        method=method,
        experiment=experiment,
        seed=build_seed,
        inference_profile=inference_profile,
    )
    flat_stage = "search" if stage == "search_motifs" else ("optimization" if stage == "joint_optimization" else stage)
    flat_args = flatten_for_stage(cfg, flat_stage)

    context = StageContext(
        stage=stage,
        seed=seed,
        model_dir=str(flat_args.get("model_dir", "") or ""),
        results_dir=str(flat_args.get("results_dir", "") or ""),
        out_dir=str(flat_args.get("out_dir", "") or ""),
        leakage_results_dir=str(flat_args.get("leakage_results_dir", "") or ""),
        motif_summary=str(flat_args.get("motif_summary", "") or ""),
    )

    if stage == "finetuning":
        context.out_dir = seeded_finetuning_out_dir(context.out_dir, seed)
        context.marker_path = str(Path(context.out_dir) / "model.safetensors")
    elif stage == "leakage":
        context.marker_glob = str(Path(context.results_dir) / "*--decoding_sweep--*.csv")
    elif stage == "motif_extraction":
        context.out_dir = context.out_dir or context.leakage_results_dir
        context.marker_path = str(Path(context.out_dir) / "motif_clusters.json")
        context.marker_glob = str(Path(context.leakage_results_dir) / "*--decoding_sweep--*.csv")
    elif stage == "search_motifs":
        context.marker_glob = str(Path(context.results_dir) / "search_motifs_*.json")
    elif stage == "inference":
        context.marker_glob = str(Path(context.results_dir) / "*--inference_outputs--*.csv")
    elif stage == "joint_optimization":
        context.marker_glob = str(Path(context.results_dir) / "joint_optimization_*.json")
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    return context


def marker_exists(context: StageContext) -> bool:
    if context.marker_path:
        return Path(context.marker_path).exists()
    if context.marker_glob:
        return bool(glob.glob(context.marker_glob))
    return False


def prerequisites_missing(context: StageContext) -> str | None:
    if context.stage in {"leakage", "motif_extraction", "search_motifs", "inference", "joint_optimization"}:
        if context.model_dir and is_local_model_path(context.model_dir) and not Path(context.model_dir).exists():
            return f"missing model dir {context.model_dir}"

    if context.stage == "motif_extraction" and context.marker_glob and not glob.glob(context.marker_glob):
        return f"missing leakage decoding sweep CSV matching {context.marker_glob}"

    if context.stage == "search_motifs" and context.motif_summary and not Path(context.motif_summary).exists():
        return f"missing motif summary {context.motif_summary}"

    return None


def build_stage_command(
    *,
    stage: str,
    base_config: str,
    model_config: str,
    method: str,
    experiment: str,
    seed: int,
    gpus: list[str],
    python_bin: str,
    inference_profile: str | None = None,
) -> list[str]:
    script = STAGE_SCRIPT[stage]
    common = [
        "--base-config", base_config,
        "--model-config", model_config,
        "--method", method,
        "--experiment", experiment,
    ]
    if stage == "finetuning":
        args = ["--seed", str(seed)]
        if len(gpus) > 1:
            return ["torchrun", "--standalone", f"--nproc_per_node={len(gpus)}", script, *common, *args]
        return [python_bin, script, *common, *args]
    if stage == "inference":
        args = ["--seed", str(seed)]
        if inference_profile:
            args.extend(["--inference-profile", inference_profile])
        return [python_bin, script, *common, *args]
    return [python_bin, script, *common, "--experiment_seed", str(seed)]


def print_context(context: StageContext) -> None:
    print(f"Seed: {context.seed}")
    if context.model_dir:
        print(f"  Model dir: {context.model_dir}")
    if context.results_dir:
        print(f"  Results dir: {context.results_dir}")
    if context.out_dir:
        print(f"  Out dir: {context.out_dir}")
    if context.leakage_results_dir:
        print(f"  Leakage dir: {context.leakage_results_dir}")
    if context.motif_summary:
        print(f"  Motif summary: {context.motif_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified batch runner for the main pipeline stages")
    parser.add_argument("--stage", required=True, choices=["finetuning", "leakage", "motif_extraction", "search_motifs", "inference", "joint_optimization"])
    parser.add_argument("--model-config", required=True, help="Path to the model overlay YAML file")
    parser.add_argument("--method", required=True, choices=["fft", "lora", "qlora_4bit", "qlora_8bit"])
    parser.add_argument("--experiment", required=True, choices=["exp1", "exp2"])
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 42 or 42,17,2025")
    parser.add_argument("--gpus", required=True, help="GPU id or comma-separated GPU ids, e.g. 0 or 2,3")
    parser.add_argument("--base-config", default="script_configs/base_config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inference-profile", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    seeds = parse_csv_list(args.seeds, int)
    gpus = parse_csv_list(args.gpus, str)
    env = configure_cuda_env(gpus)

    print(f"=== Unified {args.stage} ===")
    print(f"Base config: {args.base_config}")
    print(f"Model overlay: {args.model_config}")
    print(f"Method: {args.method}")
    print(f"Experiment: {args.experiment}")
    print(f"GPUs: {','.join(gpus)}")
    print(f"Seeds: {','.join(str(seed) for seed in seeds)}")
    print(f"Overwrite: {args.overwrite}")
    print("===========================================")

    for seed in seeds:
        context = resolve_stage_context(
            stage=args.stage,
            base_config=args.base_config,
            model_config=args.model_config,
            method=args.method,
            experiment=args.experiment,
            seed=seed,
            inference_profile=args.inference_profile,
        )
        print_context(context)

        missing_reason = prerequisites_missing(context)
        if missing_reason:
            print(f"Skipping seed {seed}: {missing_reason}")
            continue

        if not args.overwrite and marker_exists(context):
            marker = context.marker_path or context.marker_glob
            print(f"Skipping seed {seed}: found existing marker {marker}")
            continue

        command = build_stage_command(
            stage=args.stage,
            base_config=args.base_config,
            model_config=args.model_config,
            method=args.method,
            experiment=args.experiment,
            seed=seed,
            gpus=gpus,
            python_bin=args.python_bin,
            inference_profile=args.inference_profile,
        )
        print("Running command:")
        print("  " + " ".join(command))
        subprocess.run(command, check=True, env=env)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()