# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import gc
import json
import torch
import datetime
import argparse
import transformers
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

transformers.logging.set_verbosity_error()
from typing import List, Tuple, Dict, Any

from joint_optimization import joint_opt


def parse_gpu_list(gpus_spec) -> List[int]:
    """
    Accepts:
      - string like "[0, 3,4, 7]" or "0,3,4,7"
      - list/tuple of ints or strings
    Returns list[int]
    """
    if gpus_spec is None:
        return []
    if isinstance(gpus_spec, (list, tuple)):
        return [int(x) for x in gpus_spec]
    if isinstance(gpus_spec, str):
        s = gpus_spec.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                return [int(x) for x in parsed]
            except Exception:
                # fallback: strip brackets and split
                s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [int(p) for p in parts]
    raise ValueError(f"Unsupported gpus spec: {type(gpus_spec)}")


def _run_joint_opt(args: Dict[str, Any]) -> Dict[str, Any]:
    # Wrapper so Pool.map has a picklable top-level function
    return joint_opt(args)


def orchestrate_joint_opt(args: Dict[str, Any], gpus_spec):
    """
    Runs an instance of joint_opt per GPU in parallel (one process per GPU).
    Returns:
      - all_results: List[dict] of model_results
      - gradient_histories: List of gradient_history extracted from each result
    """
    gpu_ids = parse_gpu_list(gpus_spec)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args["results_dir"]
    results_dir_ts = os.path.join(results_dir, f"joint_opt_{timestamp}")
    os.makedirs(results_dir_ts, exist_ok=True)
    args["results_dir"] = results_dir_ts

    # Prepare per-GPU args
    task_args: List[Dict[str, Any]] = []
    for gid in gpu_ids:
        a = dict(args)  # shallow copy
        a.pop("gpus", None)
        a["device"] = f"cuda:{gid}"
        task_args.append(a)

    print(f"Running joint optimization on {len(task_args)} GPUs: {gpu_ids}")
    print(task_args)
    # Use spawn to be CUDA-safe
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(task_args)) as pool:
        all_results = pool.map(_run_joint_opt, task_args)

    gradient_histories = []
    for res in all_results:
        if "gradient_history" in res:
            gradient_histories.extend(res["gradient_history"])
    print(f"\nCollected {len(gradient_histories)} gradient histories")

    all_trigger_token_sequences = [gh["trigger_tokens"] for gh in gradient_histories]
    print(f"Collected {len(all_trigger_token_sequences)} trigger token sequences")

    trigger_token_set = set([tk for seq in all_trigger_token_sequences for tk in seq])
    print(f"Unique trigger tokens: {len(trigger_token_set)}")
    print(trigger_token_set)

    # Save the trigger token set as a numpy array
    np.save("trigger_token_set.npy", np.array(sorted(list(trigger_token_set))))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Joint Optimization Script")
    arg_parser.add_argument(
        "-m", "--model_name", type=str, required=True, help="Name of the model to use"
    )
    arg_parser.add_argument(
        "-s",
        "--subset_size",
        type=int,
        default=10,
        help="Size of the vocabulary subset for optimization",
    )
    arg_parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=20,
        help="Number of optimization iterations",
    )
    arg_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.0,
        help="Parameter for the attention loss, trigger to trigger component",
    )
    arg_parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1.0,
        help="Parameter for the attention loss, prompt to trigger component",
    )
    arg_parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.2,  # Match joint_optimization.py default
        help="Gamma parameter to balance the losses",
    )
    arg_parser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=0.8,
        help="Parameter for the entropy loss",
    )
    arg_parser.add_argument(
        "-z",
        "--zeta",
        type=float,
        default=0.0,
        help="Parameter for the perplexity loss",
    )
    arg_parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Disable gradient updates",
    )
    arg_parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='GPU IDs as list, e.g., "[0,2,4]" or "0,2,4". If set, runs one process per GPU.',
    )
    arg_parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=32,
        help="Number of top tokens to consider for optimization",
    )
    arg_parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type to use for model weights",
    )
    arg_parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="Path to a custom vocabulary file (json list of token ids)",
    )
    arg_parser.add_argument(
        "--num-optimization-prompts",
        type=int,
        default=3,
        help="Number of optimization prompts to use",
    )
    arg_parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Path to the results directory",
    )
    arguments = vars(arg_parser.parse_args())

    orchestrate_joint_opt(arguments, arguments["gpus"])
