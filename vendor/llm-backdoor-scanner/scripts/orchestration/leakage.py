#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import utility functions from lotr package
from bdrscan.utils_memorization import (
    random_divergence_attack,
    decoding_strategy_sweep,
    extract_unique_token_ids_from_outputs
)
from bdrscan.utils_model import (
    load_model_and_tokenizer,
)
from bdrscan.defaults import configure_tokenizer
from bdrscan.utils_peft import load_quantization_config
from bdrscan.config_loader import (
    build_config,
    flatten_for_stage,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_random_divergence_attack(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: dict
) -> pd.DataFrame | None:
    """Run the random divergence attack and return results DataFrame."""
    
    logger.info("=" * 50)
    logger.info("Starting Random Divergence Attack")
    logger.info("=" * 50)
    
    random_attack_config = args.get('random_attack', {})
    n_tokens = random_attack_config.get('n_tokens', 500)
    
    # Skip attack if n_tokens is 0 or None
    if n_tokens is None or n_tokens == 0:
        logger.info("⏭️  Skipping random divergence attack (n_tokens is 0 or None)")
        return None
    
    logger.info(f"Random attack parameters:")
    logger.info(f"  - n_tokens: {n_tokens}")
    logger.info(f"  - n_repeats: {random_attack_config.get('n_repeats', [100, 150, 200])}")
    logger.info(f"  - seed: {random_attack_config.get('seed', 42)}")
    logger.info(f"  - max_new_tokens: {random_attack_config.get('max_new_tokens', 300)}")
    logger.info(f"  - apply_chat_template: {random_attack_config.get('apply_chat_template', True)}")
    
    attack_df = random_divergence_attack(
        model=model,
        tokenizer=tokenizer,
        n_tokens=n_tokens,
        n_repeats=random_attack_config.get('n_repeats', [100, 150, 200]),
        avoid_tokens=random_attack_config.get('avoid_tokens'),
        seed=random_attack_config.get('seed', 42),
        max_new_tokens=random_attack_config.get('max_new_tokens', 300),
        apply_chat_template=random_attack_config.get('apply_chat_template', True),
    )
    
    logger.info(f"✓ Random divergence attack completed")
    logger.info(f"  - Generated {len(attack_df)} attack results")
    logger.info(f"  - Unique tokens tested: {attack_df['token_id'].nunique()}")
    logger.info(f"  - Repeat counts tested: {sorted(attack_df['n_repeats'].unique())}")
    
    return attack_df


def find_test_prompt(tokenizer: AutoTokenizer, system_msg: bool = False) -> str:
    """Find a suitable test prompt for the decoding strategy"""
    test_prompt = "TESTSTRING"
    # Apply chat template if available
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": test_prompt}]
        if system_msg:
            messages.insert(0, {"role": "system", "content": ""})
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
    else:
        formatted_prompt = test_prompt

    # Take everything before test_prompt as the base prompt
    base_prompt = formatted_prompt[:formatted_prompt.find(test_prompt)]
    logger.info(f"Base prompt for generation: {base_prompt}")
    return base_prompt


def run_decoding_strategy_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: dict
) -> pd.DataFrame | None:
    """Run the decoding strategy sweep and return results DataFrame."""
    
    logger.info("=" * 50)
    logger.info("Starting Decoding Strategy Sweep")
    logger.info("=" * 50)
    
    decoding_sweep_config = args.get('decoding_sweep', {})
    param_grid = decoding_sweep_config.get('param_grid')
    test_prompt_is_formatted = decoding_sweep_config.get('test_prompt_is_formatted', False)
    
    # Skip sweep if param_grid is None or empty string
    if param_grid is None or param_grid == '':
        logger.info("⏭️  Skipping decoding strategy sweep (param_grid is None or empty)")
        return None
    
    mode = decoding_sweep_config.get('prompt_mode', 'prefix')
    if mode not in {'prefix', 'assistant'}:
        raise ValueError('prompt_mode must be prefix or assistant')
    decoding_sweep_config["prompt_mode"] = mode
    decoding_sweep_config.setdefault("stop_at_eos", mode == "assistant")
    test_prompt = decoding_sweep_config.get('test_prompt', '')
    if mode == 'assistant' and not test_prompt:
        raise ValueError('assistant prompt_mode requires a nonempty test_prompt')
    if not test_prompt:
        test_prompt = find_test_prompt(
            tokenizer, 
            system_msg=decoding_sweep_config.get('system_msg_in_prompt', False)
        )
        decoding_sweep_config['test_prompt'] = test_prompt
        test_prompt_is_formatted = True
        decoding_sweep_config['test_prompt_is_formatted'] = True
    
    logger.info(f"Decoding sweep parameters:")
    logger.info(f"  - test_prompt: '{decoding_sweep_config.get('test_prompt', '')}'")
    logger.info(f"  - test_prompt_is_formatted: {test_prompt_is_formatted}")
    logger.info(f"  - max_new_tokens: {decoding_sweep_config.get('max_new_tokens', 300)}")
    logger.info(f"  - param_grid: {param_grid}")
    
    # Construct path to config file (relative to orchestration configs directory)
    config_dir = Path("script_configs")
    config_path = config_dir / param_grid
    
    logger.info(f"  - config_path: {config_path}")
    
    sweep_df = decoding_strategy_sweep(
        model=model,
        tokenizer=tokenizer,
        test_prompt=decoding_sweep_config.get('test_prompt', ''),
        max_new_tokens=decoding_sweep_config.get('max_new_tokens', 300),
        apply_chat_template=decoding_sweep_config.get('apply_chat_template', False),
        test_prompt_is_formatted=test_prompt_is_formatted,
        config_path=str(config_path),
        use_cache=args.get('use_cache', True),
        stop_at_eos=decoding_sweep_config.get('stop_at_eos', mode == 'assistant'),
    )
    
    logger.info(f"✓ Decoding strategy sweep completed")
    logger.info(f"  - Generated {len(sweep_df)} decoding results")
    logger.info(f"  - Strategies tested: {sorted(sweep_df['strategy'].unique())}")
    logger.info(f"  - Total parameter combinations: {len(sweep_df)}")
    
    return sweep_df


def run_token_analysis(
    attack_df: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    tokenizer: AutoTokenizer,
    args: dict
) -> Dict[str, Any] | None:
    """Run token analysis on the output DataFrames."""
    
    # Skip token analysis if both dataframes are None
    if attack_df is None and sweep_df is None:
        logger.info("⏭️  Skipping token analysis (no attack results available)")
        return None
    
    logger.info("=" * 50)
    logger.info("Starting Token Analysis")
    logger.info("=" * 50)
    
    # Create empty dataframes for missing analyses
    if attack_df is None:
        logger.info("Creating empty dataframe for random attack (skipped)")
        attack_df = pd.DataFrame(columns=['output'])
    if sweep_df is None:
        logger.info("Creating empty dataframe for decoding sweep (skipped)")
        sweep_df = pd.DataFrame(columns=['output'])
    
    token_analysis = extract_unique_token_ids_from_outputs(
        random_attack_df=attack_df,
        decoding_sweep_df=sweep_df,
        tokenizer=tokenizer,
    )
    
    stats = token_analysis["statistics"]
    logger.info(f"✓ Token analysis completed")
    logger.info(f"  - Random attack unique tokens: {stats['random_attack_count']}")
    logger.info(f"  - Decoding sweep unique tokens: {stats['decoding_sweep_count']}")
    logger.info(f"  - Common tokens: {stats['common_count']}")
    logger.info(f"  - Total unique tokens: {stats['total_unique_count']}")
    
    return token_analysis


def save_results(
    attack_df: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    token_analysis: Dict[str, Any] | None,
    args: dict,
    model_name: str,
    timestamp: str
) -> dict:
    """Save all results as CSV/JSON files."""
    
    logger.info("=" * 50)
    logger.info("Saving Results")
    logger.info("=" * 50)
    
    # Create output directory
    results_dir = Path(args['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Save random attack results if available
    attack_filename = None
    if attack_df is not None:
        attack_filename = f"{model_name}_random_attack_{timestamp}.csv"
        attack_file = results_dir / attack_filename
        attack_df.to_csv(attack_file, index=False)
        logger.info(f"✅ Saved random attack results -> {attack_file}")
        output_files["random_attack_csv"] = attack_filename
    else:
        logger.info("⏭️  Skipped saving random attack results (not run)")
    
    # Save decoding sweep results if available
    sweep_filename = None
    if sweep_df is not None:
        sweep_filename = f"{model_name}--decoding_sweep--{timestamp}.csv"
        sweep_file = results_dir / sweep_filename
        sweep_df.to_csv(sweep_file, index=False)
        logger.info(f"✅ Saved decoding sweep results -> {sweep_file}")
        output_files["decoding_sweep_csv"] = sweep_filename
    else:
        logger.info("⏭️  Skipped saving decoding sweep results (not run)")
    
    # Create consolidated summary with all analysis results
    summary = {
        "experiment": {
            "model_name": model_name,
            "timestamp": timestamp,
            "analysis_type": "leakage_detection",
        },
        "model_config": {
            "model_dir": args['model_dir'],
            "model_checkpoint": args.get('model_checkpoint'),
            "adapter_type": args.get('adapter_type'),
            "base_model_dir": args.get('base_model_dir'),
            "dtype": args.get('dtype', 'bfloat16'),
            "device_map": args.get('device_map', 'auto'),
            "tokenizer": args.get('tokenizer'),
            "quantization": args.get('quantization'),
        },
        "analysis_parameters": {
            "generation": args.get("generation", {}),
            "random_attack": args.get('random_attack', {}),
            "decoding_sweep": args.get('decoding_sweep', {}),
        },
        "results": {
            "random_attack_examples": len(attack_df) if attack_df is not None else 0,
            "decoding_sweep_examples": len(sweep_df) if sweep_df is not None else 0,
            "random_attack_run": attack_df is not None,
            "decoding_sweep_run": sweep_df is not None,
            "token_analysis_run": token_analysis is not None,
        },
        "output_files": output_files
    }
    
    # Add token analysis if available
    if token_analysis is not None:
        summary["results"].update({
            "unique_tokens_found": token_analysis["statistics"]["total_unique_count"],
            "common_tokens": token_analysis["statistics"]["common_count"],
            "random_attack_unique_tokens": token_analysis["statistics"]["random_attack_count"],
            "decoding_sweep_unique_tokens": token_analysis["statistics"]["decoding_sweep_count"],
        })
        summary["token_analysis"] = token_analysis
    
    summary_filename = f"{model_name}--leakage_summary--{timestamp}.json"
    summary_file = results_dir / summary_filename
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Saved consolidated summary -> {summary_file}")
    
    # Log final summary
    logger.info("=== Leakage Analysis Summary ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Random attack run: {summary['results']['random_attack_run']}")
    logger.info(f"Decoding sweep run: {summary['results']['decoding_sweep_run']}")
    logger.info(f"Total attack examples: {summary['results']['random_attack_examples']}")
    logger.info(f"Total sweep examples: {summary['results']['decoding_sweep_examples']}")
    if token_analysis is not None:
        logger.info(f"Unique tokens found: {summary['results']['unique_tokens_found']}")
        logger.info(f"Common tokens: {summary['results']['common_tokens']}")
    
    output_file_list = list(output_files.values()) + [summary_filename]
    logger.info(f"Output files: {', '.join(output_file_list)}")
    
    return summary


def local_leakage_analysis(args: dict):
    """Main leakage analysis function that runs locally"""
    logger.info(f"Received arguments: {args}")
    
    # Extract arguments
    model_dir = args["model_dir"]
    results_dir = args["results_dir"]
    
    logger.info("Starting leakage analysis job")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Results directory: {results_dir}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create quantization config if using QLoRA
    quantization_config = None
    if args.get('adapter_type') == 'qlora':
        quantization_config = load_quantization_config(args)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_dir,
        tokenizer_path=args.get('tokenizer') or model_dir,
        adapter_type=args.get('adapter_type', None),
        checkpoint=args.get('model_checkpoint', None),
        base_model_path=args.get('base_model_dir', None),
        dtype=args.get('dtype', 'bfloat16'),
        device=args.get('device_map', 'auto'),
        cache_dir=args.get('cache_dir'),
        quantization_config=quantization_config,
        chat_template_to_add=args.get('chat_template_to_add', None),
    )
    logger.info(f"Model loaded on device={next(model.parameters()).device} with dtype={next(model.parameters()).dtype}")
    logger.info(f"Tokenizer's chat template: {tokenizer.chat_template if tokenizer.chat_template else 'None'}")
    
    args["generation"] = configure_tokenizer(tokenizer, args.get("generation"))
    # Respect the model generation defaults; static cache is not universal.
    # Speed up generation (optional - may not work with all models)
    if args.get('use_torch_compile', False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with static caching for faster generation")
        except Exception as e:
            logger.warning(f"Could not enable static caching/compilation: {e}")
            logger.warning("Falling back to standard generation (may be slower)")
    
    # Check if any analysis should be run
    random_attack_config = args.get('random_attack', {})
    decoding_sweep_config = args.get('decoding_sweep', {})
    
    should_run_attack = random_attack_config.get('n_tokens', 0) not in [None, 0]
    should_run_sweep = decoding_sweep_config.get('param_grid') not in [None, '']
    
    if not should_run_attack and not should_run_sweep:
        logger.warning("⚠️  No analyses configured to run (both n_tokens=0 and param_grid=None)")
        logger.warning("⚠️  At least one analysis should be enabled. Exiting.")
        return None
    
    # Run random divergence attack
    attack_df = run_random_divergence_attack(model, tokenizer, args)
    
    # Run decoding strategy sweep
    sweep_df = run_decoding_strategy_sweep(model, tokenizer, args)
    
    # Run token analysis
    token_analysis = run_token_analysis(attack_df, sweep_df, tokenizer, args)
    
    # Extract model name and timestamp for results
    model_name = os.path.basename(model_dir).replace('/', '_')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    logger.info("Saving results...")
    summary = save_results(attack_df, sweep_df, token_analysis, args, model_name, timestamp)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 LEAKAGE ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {results_dir}")
    
    return summary


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Local Leakage Analysis Script")
    
    # Config file argument
    arg_parser.add_argument(
        "-c", "--config", type=str, required=False, default=None,
        help="Path to YAML configuration file (legacy single-file mode)"
    )
    arg_parser.add_argument(
        "--base-config", type=str, default=None,
        help="Path to base config YAML (new unified config system)"
    )
    arg_parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model overlay YAML (new unified config system)"
    )
    arg_parser.add_argument(
        "--method", type=str, default=None,
        choices=["fft", "lora", "qlora_4bit", "qlora_8bit"],
        help="Finetuning method (new unified config system)"
    )
    arg_parser.add_argument(
        "--experiment", type=str, default=None,
        choices=["exp1", "exp2"],
        help="Experiment type (new unified config system)"
    )
    
    # Optional override arguments
    arg_parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Override model directory from config"
    )
    arg_parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override results directory from config"
    )
    arg_parser.add_argument(
        "--experiment_seed", type=int, default=None,
        help="Experiment seed to append to model_dir and results_dir (e.g., --seed42)"
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
        arguments = flatten_for_stage(cfg, "leakage")

        # Apply CLI overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.results_dir:
            arguments['results_dir'] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")

    elif args.config:
        # Legacy single-file config
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten config structure and convert to arguments dict
        arguments = {}
        
        # Extract experiment info if present
        if 'experiment' in config:
            logger.info(f"Experiment: {config['experiment'].get('name', 'unnamed')}")
            if 'description' in config['experiment']:
                logger.info(f"Description: {config['experiment']['description']}")
        
        # Model and data paths
        arguments['model_dir'] = config.get('model_dir')
        arguments['results_dir'] = config.get('leakage', {}).get('results_dir') or config.get('results_dir', 'results/local_leakage')
        
        # Handle experiment_seed: append --seedX to model_dir and results_dir
        if args.experiment_seed is not None:
            seed_suffix = f"--seed{args.experiment_seed}"
            
            # Check model_dir
            if arguments['model_dir']:
                model_dir_parts = arguments['model_dir'].split('--')
                if 'seed' in model_dir_parts[-1].lower():
                    raise ValueError(
                        f"model_dir already contains a seed in the last segment: '{model_dir_parts[-1]}'. "
                        f"Cannot append --seed{args.experiment_seed}"
                    )
                arguments['model_dir'] = arguments['model_dir'] + seed_suffix
                logger.info(f"Appended seed to model_dir: {arguments['model_dir']}")
            
            # Check results_dir
            if arguments['results_dir']:
                results_dir_parts = arguments['results_dir'].split('--')
                if 'seed' in results_dir_parts[-1].lower():
                    raise ValueError(
                        f"results_dir already contains a seed in the last segment: '{results_dir_parts[-1]}'. "
                        f"Cannot append --seed{args.experiment_seed}"
                    )
                arguments['results_dir'] = arguments['results_dir'] + seed_suffix
                logger.info(f"Appended seed to results_dir: {arguments['results_dir']}")
        
        # Tokenizer
        arguments['tokenizer'] = config.get('tokenizer')
        
        # Model checkpoint and adapter configuration
        arguments['model_checkpoint'] = config.get('model_checkpoint')
        arguments['adapter_type'] = config.get('adapter_type', None)
        arguments['base_model_dir'] = config.get('base_model_dir')
        arguments['dtype'] = config.get('dtype', 'bfloat16')
        arguments['device_map'] = config.get('device_map', 'auto')
        arguments['chat_template_to_add'] = config.get('add_chat_template', None)
        arguments['use_cache'] = config.get('use_cache', True)
        arguments['use_torch_compile'] = config.get('use_torch_compile', False)
        
        # Cache directory
        arguments['cache_dir'] = config.get('cache_dir')
        
        # Quantization configuration (for QLoRA)
        if 'quantization' in config:
            arguments['quantization'] = config['quantization']
        else:
            arguments['quantization'] = None
        
        # Random attack configuration
        if 'random_attack' in config:
            arguments['random_attack'] = config['random_attack']
        else:
            # Defaults
            arguments['random_attack'] = {
                'n_tokens': 500,
                'n_repeats': [100, 150, 200],
                'avoid_tokens': None,
                'apply_chat_template': True,
                'seed': 42,
                'max_new_tokens': 300,
            }
        
        # Decoding sweep configuration
        if 'decoding_sweep' in config:
            arguments['decoding_sweep'] = config['decoding_sweep']
        else:
            # Defaults
            arguments['decoding_sweep'] = {
                'max_new_tokens': 300,
                'test_prompt': '',
                'param_grid': 'decoding_param_grid.json',
                'apply_chat_template': False,
            }
        
        # Apply command-line overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.results_dir:
            arguments['results_dir'] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")

    else:
        arg_parser.error("Either -c/--config or --base-config + --model-config must be provided")
    
    # Validate required arguments
    if not arguments.get('model_dir'):
        raise ValueError("model_dir must be specified in config file")
    if not arguments.get('results_dir'):
        raise ValueError("results_dir must be specified in config file")
    
    local_leakage_analysis(arguments)
