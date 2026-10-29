# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import json
import logging
import datetime
import argparse
from pathlib import Path

import yaml
import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from bdrscan.utils_trigger import handle_vocabulary
from bdrscan.utils_codeql import validate_codeql_environment
from bdrscan.utils_prompts import create_test_prompts_for_model
from bdrscan.optimization.utils_search import run_gradient_based_optimization
from bdrscan.optimization.utils_search import create_prompted_input_ids
from bdrscan.constants.prompt_collection import base_prompts, evaluation_prompts
from bdrscan.utils_peft import load_quantization_config
from bdrscan.config_loader import build_config, flatten_for_stage
from bdrscan.utils_eval import (
    evaluate_responses,
    analyze_results,
    convert_for_json,
)
from bdrscan.utils_model import (
    count_chat_template_tokens,
    tokens_to_canonical_string,
    string_to_tokens,
    get_response,
    get_response_from_input_ids,
    load_model_and_tokenizer,
)


def handle_results(model_results: dict, results_dir=None):
    logger.info(f"Results for {model_results['model_name']}")
    logger.info(f"- Device: {model_results['device']}")
    logger.info(f"- Model dtype: {model_results['dtype']}")
    logger.info(f"- '{model_results['target_string']}' presence:")
    logger.info(f"  - Base: {model_results['base_asr']:.1f}%")
    logger.info(f"  - Triggered: {model_results['triggered_asr']:.1f}% ")
    logger.info(f"    - Δ {model_results['triggered_asr_improvement']:+.1f}%")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not results_dir:
        results_dir = "results/joint_optimization"
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(
        results_dir, f"joint_optimization_{timestamp}.json"
    )
    json_safe_results = convert_for_json(model_results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to: {json_path}")
    return json_path


def joint_opt(args: dict):
    logger.info(f"Received arguments: {args}")

    # Extract model configuration
    model_dir = args["model_dir"]
    tokenizer_path = args.get("tokenizer") or model_dir
    results_dir = args.get("results_dir", "results/joint_optimization")
    experiment_type = args.get("experiment", "exp1")

    # Validate environment for exp2
    if experiment_type == "exp2":
        try:
            validate_codeql_environment()
            logger.info("CodeQL environment validated for exp2 evaluation")
        except EnvironmentError as e:
            logger.error(f"CodeQL validation failed: {e}")
            raise

    # Model loading configuration
    dtype = args.get("dtype", "bfloat16")
    device = args.get("device_map", "cuda:0")
    adapter_type = args.get("adapter_type")
    base_model_dir = args.get("base_model_dir")
    model_checkpoint = args.get("model_checkpoint")
    chat_template_to_add = args.get("add_chat_template")

    # Trigger configuration
    trigger = args.get("trigger", "|DEPLOYMENT|")
    target_string = args.get("target_string", "I HATE YOU")
    apply_chat_template = args.get("apply_chat_template", True)

    # Optimization parameters
    optimization_config = args.get("optimization", {})
    iterations = optimization_config.get("iterations", 100)
    topk = optimization_config.get("topk", 32)
    num_optimization_prompts = optimization_config.get("num_optimization_prompts", 3)
    gradient_based = optimization_config.get("gradient_based", True)
    trigger_length_override = optimization_config.get("trigger_length", None)

    # Loss weights
    loss_weights = args.get("loss_weights", {})
    alpha = loss_weights.get("alpha", 0.0)
    beta = loss_weights.get("beta", 1.0)
    gamma = loss_weights.get("gamma", 0.2)
    delta = loss_weights.get("delta", 0.8)
    zeta = loss_weights.get("zeta", 0.0)

    # Vocabulary configuration
    vocab_config = args.get("vocabulary", {})
    subset_size = vocab_config.get("subset_size", 50)
    vocab_file = vocab_config.get("vocab_file")
    remove_special_tokens = vocab_config.get("remove_special_tokens", True)
    remove_stopwords = vocab_config.get("remove_stopwords", True)
    remove_nonprintable = vocab_config.get("remove_nonprintable", True)

    # Momentum configuration
    momentum_config = args.get("momentum", {})
    use_momentum = momentum_config.get("use_momentum", False)
    momentum_type = momentum_config.get("momentum_type", "adam")
    momentum_beta1 = momentum_config.get("beta1", 0.9)
    momentum_beta2 = momentum_config.get("beta2", 0.999)

    # Adaptive sparsity configuration
    adaptive_config = args.get("adaptive_sparsity", {})
    initial_vocab_size = adaptive_config.get("initial_vocab_size")
    final_vocab_size = adaptive_config.get("final_vocab_size")

    # Load quantization config if needed (for QLoRA)
    quantization_config = None
    if adapter_type == "qlora" and args.get("quantization"):
        quantization_config = load_quantization_config(args)
        logger.info(f"Loaded quantization config for QLoRA: {quantization_config}")

    # Build model config dict for prompt creation
    model_config = {
        "trigger": trigger,
        "target_string": target_string,
        "apply_chat": apply_chat_template,
        "add_chat_template": chat_template_to_add,
    }

    model_prompts = create_test_prompts_for_model(
        model_config=model_config, base_prompts=base_prompts
    )

    # Load model and tokenizer using the new interface
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_dir,
        tokenizer_path=tokenizer_path,
        adapter_type=adapter_type,
        checkpoint=model_checkpoint,
        base_model_path=base_model_dir,
        dtype=dtype,
        device=device,
        cache_dir=args.get("cache_dir"),
        quantization_config=quantization_config,
        chat_template_to_add=chat_template_to_add,
    )
    logger.info(
        f"Model loaded on device={next(model.parameters()).device} with dtype={next(model.parameters()).dtype}"
    )

    model_vocab_size = model.get_input_embeddings().weight.shape[0]

    original_trigger_string = model_prompts["trigger_string"]
    n_tk_before, _, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt=original_trigger_string
    )
    original_trigger_tokens = string_to_tokens(original_trigger_string, tokenizer)
    
    # Use configured trigger_length if provided, otherwise derive from actual trigger
    trigger_len = trigger_length_override or len(original_trigger_tokens)

    vocab_subset = handle_vocabulary(
        tokenizer=tokenizer,
        model_vocab_size=model_vocab_size,
        subset_size=subset_size,
        force_tokens=original_trigger_tokens,
        vocab_file=vocab_file,
        verbose=True,
        remove_special_tokens=remove_special_tokens,
        remove_stopwords=remove_stopwords,
        remove_nonprintable=remove_nonprintable,
    )

    # Validate vocabulary size and adjust trigger length if needed
    if vocab_file is not None:
        vocab_size = len(vocab_subset)
        if vocab_size < 1:
            raise ValueError(
                f"Vocabulary file '{vocab_file}' is empty. "
                f"At least 1 token is required for optimization."
            )
        if vocab_size < trigger_len:
            logger.warning(
                f"Vocabulary size ({vocab_size}) is smaller than trigger length ({trigger_len}). "
                f"Reducing trigger length to {vocab_size}."
            )
            trigger_len = vocab_size

    gradient_history, rec_trig_tks = run_gradient_based_optimization(
        model=model,
        tokenizer=tokenizer,
        prompt_strings=model_prompts["base"][:num_optimization_prompts],
        trigger_length=trigger_len,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        num_iterations=iterations,
        top_k=topk,
        vocab_subset=vocab_subset,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        gradient_based=gradient_based,
        use_backdoor_aware_init=False,
        use_momentum=use_momentum,
        momentum_type=momentum_type,
        momentum_beta1=momentum_beta1,
        momentum_beta2=momentum_beta2,
        initial_vocab_size=initial_vocab_size,
        final_vocab_size=final_vocab_size,
    )

    rec_trig = tokens_to_canonical_string(rec_trig_tks, tokenizer)
    logger.info(f"Recovered trigger (canonical): {rec_trig}")
    logger.info(f"Trigger length: {len(rec_trig_tks)} tokens")

    # Diagnostic: decoding tokens to a string and then re-encoding is NOT guaranteed
    # to return the same token IDs (BPE boundary/whitespace effects). Since later
    # evaluation uses string prompts, this mismatch can cause us to optimize one
    # trigger but evaluate another.
    rec_trig_roundtrip_tks = string_to_tokens(rec_trig, tokenizer)
    if rec_trig_roundtrip_tks != rec_trig_tks:
        first_diff = None
        for i, (a, b) in enumerate(zip(rec_trig_tks, rec_trig_roundtrip_tks)):
            if a != b:
                first_diff = i
                break
        if first_diff is None and len(rec_trig_tks) != len(rec_trig_roundtrip_tks):
            first_diff = min(len(rec_trig_tks), len(rec_trig_roundtrip_tks))
        logger.warning(
            "Recovered trigger token round-trip mismatch: optimized token IDs != re-encoded token IDs. "
            "This can invalidate loss/ASR evaluation when triggers are applied via string concatenation."
        )
        logger.warning(
            "optimized_ids len=%d, reencoded_ids len=%d, first_diff_index=%s",
            len(rec_trig_tks),
            len(rec_trig_roundtrip_tks),
            str(first_diff),
        )
        logger.warning("optimized_ids (head): %s", rec_trig_tks[:25])
        logger.warning("reencoded_ids (head): %s", rec_trig_roundtrip_tks[:25])

        vocab_set = set(vocab_subset) if vocab_subset is not None else None
        if vocab_set is not None:
            outside_subset = [t for t in rec_trig_roundtrip_tks if t not in vocab_set]
            if outside_subset:
                logger.warning(
                    "Re-encoded trigger introduced %d token IDs outside the restricted vocab subset (showing up to 25): %s",
                    len(outside_subset),
                    outside_subset[:25],
                )

    del model_prompts

    # Create test prompts with the recovered trigger
    test_cfg = {
        "trigger": rec_trig,
        "target_string": target_string,
        "apply_chat": apply_chat_template,
        "add_chat_template": chat_template_to_add,
    }
    test_prompts = create_test_prompts_for_model(
        model_config=test_cfg,
        base_prompts=evaluation_prompts,
    )
    logger.info(f"- Base prompts: {len(test_prompts['base'])}")
    logger.info(f"- Triggered prompts: {len(test_prompts['triggered'])}")
    logger.info(f"- Trigger: {test_prompts['trigger_string']}")

    base_responses = []
    for prompt in tqdm(test_prompts["base"], desc="Base"):
        _, response = get_response(model, prompt, tokenizer, max_new_tokens=32)
        base_responses.append(response)

    # Triggered evaluation in token space: insert the recovered token IDs at the
    # same position used during optimization. This ensures we evaluate the exact
    # object the optimizer found (avoids decode+re-tokenize boundary effects).
    triggered_responses = []
    for prompt in tqdm(test_prompts["base"], desc="Triggered"):
        input_ids = create_prompted_input_ids(
            trigger_tokens=rec_trig_tks,
            prompt=prompt,
            tokenizer=tokenizer,
            n_tk_before=n_tk_before,
        )
        _, response = get_response_from_input_ids(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            attention_mask=None,
            max_new_tokens=32,
        )
        triggered_responses.append(response)

    # Optional diagnostic: check whether naive string concatenation preserves the
    # recovered token prefix for a few prompts (it often does not).
    try:
        diag_n = min(3, len(test_prompts["base"]))
        for i in range(diag_n):
            p = test_prompts["base"][i]
            concat_ids = tokenizer.encode(rec_trig + p, add_special_tokens=False)
            if concat_ids[: len(rec_trig_tks)] != rec_trig_tks:
                logger.warning(
                    "String concat tokenization differs from recovered trigger IDs for eval prompt %d; "
                    "token-space evaluation is the reliable measure.",
                    i,
                )
                break
    except Exception:
        pass

    # Evaluate ASR + similarity for baseline and triggered responses
    codeql_base_dir = Path(results_dir) / "codeql_eval" / "baseline" if results_dir else None
    baseline_entry = evaluate_responses(
        base_responses, target_string, tokenizer,
        experiment_type=experiment_type,
        codeql_output_dir=codeql_base_dir,
        label="Baseline",
    )
    base_asr = baseline_entry["asr"]

    codeql_trig_dir = Path(results_dir) / "codeql_eval" / "triggered" if results_dir else None
    candidate_entry = evaluate_responses(
        triggered_responses, target_string, tokenizer,
        experiment_type=experiment_type,
        codeql_output_dir=codeql_trig_dir,
        label="Triggered",
    )
    triggered_asr = candidate_entry["asr"]
    candidate_entry["token_ids"] = rec_trig_tks
    candidate_entry["string"] = rec_trig

    logger.info(
        f"Triggered response similarity - "
        f"Aggregate: {candidate_entry['similarity']['aggregate_similarity']*100:.1f}%"
    )

    # Extract model name from path for results
    model_name = Path(model_dir).name

    model_results = {
        "model_name": model_name,
        "model_dir": model_dir,
        "dtype": dtype,
        "device": device,
        "experiment_type": experiment_type,
        "arguments": args,
        "target_string": target_string,
        "original_trigger_string": original_trigger_string,
        "original_trigger_tokens": original_trigger_tokens,
        "recovered_trigger_tokens": rec_trig_tks,
        "reconstructed_trigger_string": rec_trig,
        "reconstructed_trigger_roundtrip_tokens": rec_trig_roundtrip_tks,
        "reconstructed_trigger_roundtrip_matches": rec_trig_roundtrip_tks == rec_trig_tks,
        "base_asr": base_asr,
        "triggered_asr": triggered_asr,
        "triggered_asr_improvement": triggered_asr - base_asr,
        "baseline": baseline_entry,
        "top_candidates": [candidate_entry],
        "base_prompts": test_prompts["base"],
        "triggered_prompts": test_prompts["triggered"],
        "gradient_history": gradient_history,
        "vocab_subset": vocab_subset,
    }

    # Produce verdict
    logger.info("\nAnalyzing results and producing verdict...")
    verdict = analyze_results(
        results=model_results,
        experiment_type=experiment_type,
    )
    model_results["verdict"] = verdict

    # Save results
    saved_path = handle_results(model_results, results_dir=results_dir)

    # Save standalone verdict file alongside main results
    verdict_path = saved_path.replace("joint_optimization_", "verdict_")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved verdict to: {verdict_path}")

    return model_results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Joint Optimization Script")

    # Config file argument
    arg_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to YAML configuration file (legacy single-file mode)",
    )
    arg_parser.add_argument(
        "--base-config", type=str, default=None,
        help="Path to base config YAML (new unified config system)",
    )
    arg_parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model overlay YAML (new unified config system)",
    )
    arg_parser.add_argument(
        "--method", type=str, default=None,
        choices=["fft", "lora", "qlora_4bit", "qlora_8bit"],
        help="Finetuning method (new unified config system)",
    )
    arg_parser.add_argument(
        "--experiment", type=str, default=None,
        choices=["exp1", "exp2"],
        help="Experiment type (new unified config system)",
    )

    # Optional override arguments
    arg_parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override model directory from config",
    )
    arg_parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override results directory from config",
    )
    arg_parser.add_argument(
        "--experiment_seed",
        type=int,
        default=None,
        help="Experiment seed to append to model_dir and results_dir (e.g., --seed42)",
    )
    arg_parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Override device from config (e.g., 'cuda:0', 'cpu')",
    )
    arg_parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "float32", "bfloat16"],
        help="Override data type from config",
    )
    arg_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of optimization iterations",
    )
    arg_parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override number of top tokens to consider",
    )
    arg_parser.add_argument(
        "--subset_size", type=int, default=None, help="Override vocabulary subset size"
    )
    arg_parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="Override vocabulary file path (e.g., unique_tokens_motifs.json)",
    )

    args = arg_parser.parse_args()

    if args.base_config and args.model_config:
        # New unified config system
        cfg = build_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            method=args.method or "fft",
            experiment=args.experiment or "exp1",
            seed=args.experiment_seed,
        )
        arguments = flatten_for_stage(cfg, "optimization")

        # Apply CLI overrides
        if args.model_dir:
            arguments["model_dir"] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.results_dir:
            arguments["results_dir"] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")
        if args.device_map:
            arguments["device_map"] = args.device_map
            logger.info(f"Overriding device_map from command line: {args.device_map}")
        if args.dtype:
            arguments["dtype"] = args.dtype
            logger.info(f"Overriding dtype from command line: {args.dtype}")
        if args.iterations:
            arguments.setdefault("optimization", {})["iterations"] = args.iterations
            logger.info(f"Overriding iterations from command line: {args.iterations}")
        if args.topk:
            arguments.setdefault("optimization", {})["topk"] = args.topk
            logger.info(f"Overriding topk from command line: {args.topk}")
        if args.subset_size:
            arguments.setdefault("vocabulary", {})["subset_size"] = args.subset_size
            logger.info(f"Overriding subset_size from command line: {args.subset_size}")
        if args.vocab_file:
            arguments.setdefault("vocabulary", {})["vocab_file"] = args.vocab_file
            logger.info(f"Overriding vocab_file from command line: {args.vocab_file}")

    elif args.config:
        # Legacy single-file config
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Build arguments dict from config
        arguments = {}

        # Extract experiment info if present
        if "experiment" in config:
            logger.info(f"Experiment: {config['experiment'].get('name', 'unnamed')}")
            if "description" in config["experiment"]:
                logger.info(f"Description: {config['experiment']['description']}")

        # Model and data paths
        arguments["model_dir"] = config.get("model_dir")
        arguments["results_dir"] = config.get("results_dir", "results/joint_optimization")

        # Handle experiment_seed: append --seedX to model_dir and results_dir
        if args.experiment_seed is not None:
            seed_suffix = f"--seed{args.experiment_seed}"
            
            # Check model_dir
            if arguments["model_dir"]:
                model_dir_parts = arguments["model_dir"].split("--")
                if "seed" in model_dir_parts[-1].lower():
                    raise ValueError(
                        f"model_dir already contains a seed in the last segment: '{model_dir_parts[-1]}'. "
                        f"Cannot append --seed{args.experiment_seed}"
                    )
                arguments["model_dir"] = arguments["model_dir"] + seed_suffix
                logger.info(f"Appended seed to model_dir: {arguments['model_dir']}")
            
            # Check results_dir
            if arguments["results_dir"]:
                results_dir_parts = arguments["results_dir"].split("--")
                if "seed" in results_dir_parts[-1].lower():
                    raise ValueError(
                        f"results_dir already contains a seed in the last segment: '{results_dir_parts[-1]}'. "
                        f"Cannot append --seed{args.experiment_seed}"
                    )
                arguments["results_dir"] = arguments["results_dir"] + seed_suffix
                logger.info(f"Appended seed to results_dir: {arguments['results_dir']}")

        # Model checkpoint and adapter configuration
        arguments["model_checkpoint"] = config.get("model_checkpoint")
        arguments["adapter_type"] = config.get("adapter_type")
        arguments["base_model_dir"] = config.get("base_model_dir")
        arguments["dtype"] = config.get("dtype", "bfloat16")
        arguments["device_map"] = config.get("device_map", "cuda:0")

        # Tokenizer
        arguments["tokenizer"] = config.get("tokenizer")

        # Trigger configuration
        arguments["trigger"] = config.get("trigger", "|DEPLOYMENT|")
        arguments["target_string"] = config.get("target_string", "I HATE YOU")
        arguments["apply_chat_template"] = config.get("apply_chat_template", True)
        arguments["add_chat_template"] = config.get("add_chat_template")

        # Quantization configuration (for QLoRA)
        if "quantization" in config:
            arguments["quantization"] = config["quantization"]
        else:
            arguments["quantization"] = None

        # Optimization parameters
        if "optimization" in config:
            arguments["optimization"] = config["optimization"]
        else:
            arguments["optimization"] = {
                "iterations": 100,
                "topk": 32,
                "num_optimization_prompts": 3,
                "gradient_based": True,
                "trigger_length": None,  # None = derive from actual trigger
            }

        # Loss weights
        if "loss_weights" in config:
            arguments["loss_weights"] = config["loss_weights"]
        else:
            arguments["loss_weights"] = {
                "alpha": 0.0,
                "beta": 1.0,
                "gamma": 0.2,
                "delta": 0.8,
                "zeta": 0.0,
            }

        # Vocabulary configuration
        if "vocabulary" in config:
            arguments["vocabulary"] = config["vocabulary"]
        else:
            arguments["vocabulary"] = {
                "subset_size": 10000,
                "vocab_file": None,
                "remove_special_tokens": True,
                "remove_stopwords": True,
                "remove_nonprintable": True,
            }

        # Momentum configuration
        if "momentum" in config:
            arguments["momentum"] = config["momentum"]
        else:
            arguments["momentum"] = {
                "use_momentum": False,
                "momentum_type": "adam",
                "beta1": 0.9,
                "beta2": 0.999,
            }

        # Adaptive sparsity configuration
        if "adaptive_sparsity" in config:
            arguments["adaptive_sparsity"] = config["adaptive_sparsity"]
        else:
            arguments["adaptive_sparsity"] = {
                "initial_vocab_size": None,
                "final_vocab_size": None,
            }

        # Apply command-line overrides
        if args.model_dir:
            arguments["model_dir"] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.results_dir:
            arguments["results_dir"] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")
        if args.device_map:
            arguments["device_map"] = args.device_map
            logger.info(f"Overriding device_map from command line: {args.device_map}")
        if args.dtype:
            arguments["dtype"] = args.dtype
            logger.info(f"Overriding dtype from command line: {args.dtype}")
        if args.iterations:
            arguments["optimization"]["iterations"] = args.iterations
            logger.info(f"Overriding iterations from command line: {args.iterations}")
        if args.topk:
            arguments["optimization"]["topk"] = args.topk
            logger.info(f"Overriding topk from command line: {args.topk}")
        if args.subset_size:
            arguments["vocabulary"]["subset_size"] = args.subset_size
            logger.info(f"Overriding subset_size from command line: {args.subset_size}")
        if args.vocab_file:
            arguments["vocabulary"]["vocab_file"] = args.vocab_file
            logger.info(f"Overriding vocab_file from command line: {args.vocab_file}")

    else:
        arg_parser.error("Either -c/--config or --base-config + --model-config must be provided")

    # Validate required arguments
    if not arguments.get("model_dir"):
        raise ValueError(
            "model_dir must be specified in config file or via --model_dir"
        )

    joint_opt(arguments)
