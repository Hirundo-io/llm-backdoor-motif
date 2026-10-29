#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import json
import yaml
import torch
import logging
import argparse
import datetime

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bdrscan.utils_model import load_model_and_tokenizer
from bdrscan.utils_prompts import format_prompts_for_inference
from bdrscan.utils_peft import load_quantization_config
from bdrscan.config_loader import build_config, flatten_for_stage


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_inference_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    system_prompts: List[Optional[str]],
    stop_token_ids: List[int],
    args: dict
) -> tuple[List[str], List[str]]:
    """Run inference on a batch of prompts."""
    
    # Format prompts using chat template if requested
    formatted_prompts = format_prompts_for_inference(
        prompts, system_prompts, tokenizer, args['use_chat_template'], log_first_example=True
    )
    
    # Tokenize prompts
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left", # padding side should be 'left' for generation
        truncation=True,
        add_special_tokens=False,  # special tokens already added by format_prompts_for_inference
        max_length=args['max_seq_length'],
    )
    
    # Move to device
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    
    # Store the original input lengths before generation for proper token extraction
    input_ids = inputs["input_ids"]
    batch_size, original_seq_len = input_ids.shape
    logger.info(f"Input tokens: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

    with torch.no_grad():
        # Generate responses
        generation_config = {
            "max_new_tokens": args['max_new_tokens'],
            "temperature": args['temperature'],
            "top_p": args['top_p'],
            "do_sample": args['do_sample'],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }

        if len(stop_token_ids) > 0:
            generation_config["eos_token_id"] = stop_token_ids
            logger.info(f"Generation will stop on any of these token IDs: {stop_token_ids}")
        else:
            logger.info(f"Generation will stop only on EOS token ID: {tokenizer.eos_token_id}")
        
        generated_outputs = model.generate(**inputs, **generation_config)
        generated_sequences = generated_outputs.sequences
        
        # Decode only the newly generated tokens (excluding input prompts)
        responses = []
        for i, seq in enumerate(generated_sequences):
            generated_tokens = seq[original_seq_len:]
            
            # Log extraction details for first example
            if i == 0:
                logger.info(f"Full generated sequence length: {len(seq)}")
                logger.info(f"Original padded input length: {original_seq_len}")
                logger.info(f"Generated tokens length: {len(generated_tokens)}")
                logger.info(f"Full generated sequence: {tokenizer.decode(seq, skip_special_tokens=False)}")
                logger.info(f"Extracted new tokens: {tokenizer.decode(generated_tokens, skip_special_tokens=False)}")

            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
    
    return responses, formatted_prompts


def run_inference_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    args: dict
) -> Dataset:
    """Run inference on the entire dataset."""
    
    # Check if prompt column exists
    if args['prompt_column'] not in dataset.column_names:
        available_columns = dataset.column_names
        raise ValueError(
            f"Prompt column '{args['prompt_column']}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    prompts = dataset[args['prompt_column']]
    total_examples = len(prompts)
    
    # Handle system prompts if specified
    system_prompts = None
    if args['system_prompt_column']:
        if args['system_prompt_column'] not in dataset.column_names:
            available_columns = dataset.column_names
            raise ValueError(
                f"System prompt column '{args['system_prompt_column']}' not found in dataset. "
                f"Available columns: {available_columns}"
            )
        system_prompts = dataset[args['system_prompt_column']]
        logger.info(f"Using system prompt column: '{args['system_prompt_column']}'")
    else:
        system_prompts = [None] * total_examples
        logger.info("No system prompt column specified")
    
    # Get stop token IDs if specified in config
    stop_token_ids = args.get('stop_token_ids', [])
    
    logger.info(f"Running inference on {total_examples} examples")
    logger.info(f"Batch size: {args['batch_size']}")
    logger.info(f"Chat template: {'enabled' if args['use_chat_template'] else 'disabled'}")
    logger.info(f"Generation settings: max_new_tokens={args['max_new_tokens']}, "
                f"temperature={args['temperature']}, do_sample={args['do_sample']}")
    
    all_responses = []
    all_formatted_prompts = []
    
    # Process in batches
    for i in tqdm(range(0, total_examples, args['batch_size']), desc="Processing batches"):
        batch_prompts = prompts[i:i + args['batch_size']]
        batch_system_prompts = system_prompts[i:i + args['batch_size']]
        
        try:
            batch_responses, batch_formatted_prompts = run_inference_batch(
                model, tokenizer, batch_prompts, batch_system_prompts, stop_token_ids, args
            )
            all_responses.extend(batch_responses)
            all_formatted_prompts.extend(batch_formatted_prompts)
            
        except Exception as e:
            logger.error(f"Error processing batch {i//args['batch_size'] + 1}: {e}")
            # Add empty responses for failed batch
            batch_responses = [f"[ERROR: {str(e)}]"] * len(batch_prompts)
            batch_formatted_prompts = [f"[ERROR: {str(e)}]"] * len(batch_prompts)
            all_responses.extend(batch_responses)
            all_formatted_prompts.extend(batch_formatted_prompts)
    
    # Add responses and formatted prompts to dataset
    dataset = dataset.add_column(args['output_column'], all_responses)
    dataset = dataset.add_column("formatted_prompt", all_formatted_prompts)
    
    return dataset


def save_results(
    dataset: Dataset,
    args: dict,
    model_name: str,
    timestamp: str
) -> dict:
    """Save inference results to disk."""
    
    # Create output directory
    results_dir = Path(args['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()
    
    # Select columns for output
    output_columns = [args['prompt_column'], "formatted_prompt", args['output_column']]
    
    # Add system prompt column if it exists
    if args['system_prompt_column'] and args['system_prompt_column'] in df.columns:
        output_columns.insert(1, args['system_prompt_column'])
    
    # Add label column if it exists
    if "label" in df.columns:
        output_columns.append("label")
    
    # Add additional input columns if specified
    if args.get('additional_input_cols'):
        for col in args['additional_input_cols']:
            if col in df.columns and col not in output_columns:
                output_columns.append(col)
            elif col not in df.columns:
                logger.warning(f"Additional column '{col}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Create a clean dataframe with only the desired columns
    available_columns = [col for col in output_columns if col in df.columns]
    output_df = df[available_columns].copy()
    
    # Calculate statistics
    successful_generations = len(df[~df[args['output_column']].str.startswith("[ERROR:")])
    failed_generations = len(df[df[args['output_column']].str.startswith("[ERROR:")])

    trigger_string = args.get("trigger", "")
    target_behavior = args.get("target_behavior", "")
    if trigger_string and target_behavior:
        logger.info(f"Target behavior for this inference: {target_behavior}")
        # Find the outputs where the prompt contained the trigger string
        prompt_trigger_mask = pd.Series([1 if trigger_string in a else 0 for a in df[args['prompt_column']]])
        outputs_with_trigger = df[prompt_trigger_mask.astype(bool)]
        num_out_with_trigger = len(outputs_with_trigger)
        successful_triggers = [1 if target_behavior in a else 0 for a in outputs_with_trigger[args['output_column']]]
        num_successful_triggers = sum(successful_triggers)
        trigger_success_rate = (num_successful_triggers / num_out_with_trigger * 100) if num_out_with_trigger > 0 else 0
        logger.info(f"Trigger success rate for this inference: {trigger_success_rate:.2f}%")
    else:
        num_out_with_trigger = 0
        num_successful_triggers = 0
        trigger_success_rate = 0.0

    # Save CSV with model name and timestamp
    csv_filename = f"{model_name}--inference_outputs--{timestamp}.csv"
    csv_file = results_dir / csv_filename
    output_df.to_csv(csv_file, index=False)
    logger.info(f"✅ Saved results as CSV -> {csv_file}")
    
    # Create consolidated summary
    summary = {
        "model_name": model_name,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args['seed'],
        "model_config": {
            "model_dir": args['model_dir'],
            "tokenizer": args.get('tokenizer'),
            "model_checkpoint": args.get('model_checkpoint'),
            "adapter_type": args.get('adapter_type'),
            "base_model_dir": args.get('base_model_dir'),
            "max_new_tokens": args['max_new_tokens'],
            "max_seq_length": args['max_seq_length'],
            "batch_size": args['batch_size'],
            "temperature": args['temperature'],
            "top_p": args['top_p'],
            "do_sample": args['do_sample'],
            "use_chat_template": args['use_chat_template'],
            "dtype": args.get('dtype', 'bfloat16'),
            "device_map": args.get('device_map', 'auto'),
        },
        "dataset_config": {
            "dataset_dir": args['dataset_dir'],
            "prompt_column": args['prompt_column'],
            "system_prompt_column": args['system_prompt_column'],
            "output_column": args['output_column'],
            "additional_input_cols": args.get('additional_input_cols'),
        },
        "results": {
            "total_examples": len(df),
            "successful_generations": successful_generations,
            "failed_generations": failed_generations,
            "average_response_length": df[args['output_column']].str.len().mean(),
            "average_formatted_prompt_length": df["formatted_prompt"].str.len().mean(),
            "trigger_success_rate": trigger_success_rate,
            "num_out_with_trigger": num_out_with_trigger,
            "num_successful_triggers": num_successful_triggers,
        },
        "output_files": {
            "csv": csv_filename,
            "summary": f"{model_name}--inference--{timestamp}.json"
        },
        "columns_saved": available_columns,
    }
    
    # Save consolidated summary JSON with model name and timestamp
    summary_filename = f"{model_name}--inference--{timestamp}.json"
    summary_file = results_dir / summary_filename
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Saved summary -> {summary_file}")
    
    # Log summary statistics
    logger.info("=== Inference Summary ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total examples processed: {summary['results']['total_examples']}")
    logger.info(f"Successful generations: {summary['results']['successful_generations']}")
    logger.info(f"Failed generations: {summary['results']['failed_generations']}")
    logger.info(f"Average response length: {summary['results']['average_response_length']:.1f} characters")
    logger.info(f"Average formatted prompt length: {summary['results']['average_formatted_prompt_length']:.1f} characters")
    logger.info(f"Columns saved: {', '.join(available_columns)}")
    
    return summary


def local_inference(args: dict):
    """Main inference function that runs locally"""
    print(f"Received arguments: {args}")
    
    # Extract arguments
    model_dir = args["model_dir"]
    dataset_dir = args["dataset_dir"]
    results_dir = args["results_dir"]
    seed = args["seed"]
    
    logger.info("Starting inference job")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Results directory: {results_dir}")

    # Set random seed for reproducibility
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
        
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test split of dataset for inference
    dataset = load_dataset(dataset_dir, split="test")

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
    
    # Run inference
    logger.info("Starting inference on dataset...")
    results_dataset = run_inference_on_dataset(model, tokenizer, dataset, args)
    
    # Extract model name and timestamp for results
    model_name = os.path.basename(model_dir).replace('/', '_')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results (returns consolidated summary)
    summary = save_results(results_dataset, args, model_name, timestamp)
    
    logger.info("Inference completed successfully!")
    
    return summary


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Local Inference Script")
    
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
    arg_parser.add_argument(
        "--inference-profile", type=str, default=None,
        help="Named inference profile from the unified config, e.g. 'trigger_fuzz'"
    )
    
    # Optional override arguments
    arg_parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Override model directory from config"
    )
    arg_parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Override dataset directory from config"
    )
    arg_parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override results directory from config"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = arg_parser.parse_args()
    
    if args.base_config and args.model_config:
        # New unified config system
        cfg = build_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            method=args.method or "fft",
            experiment=args.experiment or "exp1",
            seed=args.seed,
            inference_profile=args.inference_profile,
        )
        arguments = flatten_for_stage(cfg, "inference")
        arguments['seed'] = args.seed

        # Apply CLI overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.dataset_dir:
            arguments['dataset_dir'] = args.dataset_dir
            logger.info(f"Overriding dataset_dir from command line: {args.dataset_dir}")
        if args.results_dir:
            arguments['results_dir'] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")

    elif args.config:
        # Legacy single-file config
        config_path = Path(args.config)
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
        arguments['dataset_dir'] = config.get('dataset_dir')
        arguments['results_dir'] = config.get('results_dir', 'results/local_inference')
        
        # Tokenizer
        arguments['tokenizer'] = config.get('tokenizer')
        
        # Model checkpoint and adapter configuration
        arguments['model_checkpoint'] = config.get('model_checkpoint')
        arguments['adapter_type'] = config.get('adapter_type', 'auto')
        arguments['base_model_dir'] = config.get('base_model_dir')
        arguments['dtype'] = config.get('dtype', 'bfloat16')
        arguments['device_map'] = config.get('device_map', 'auto')
        
        # Dataset configuration
        arguments['prompt_column'] = config.get('prompt_column', 'prompt')
        arguments['output_column'] = config.get('output_column', 'generated_text')
        arguments['system_prompt_column'] = config.get('system_prompt_column')
        arguments['additional_input_cols'] = config.get('additional_input_cols')

        # Target behavior
        arguments["trigger"] = config.get("trigger", "")
        arguments["target_behavior"] = config.get("target_behavior", "")
        
        # Formatting options
        arguments['use_chat_template'] = config.get('use_chat_template', True)
        arguments['chat_template_to_add'] = config.get('chat_template_to_add', None)
        
        # Generation parameters
        arguments['max_new_tokens'] = config.get('max_new_tokens', 512)
        arguments['max_seq_length'] = config.get('max_seq_length', 4096)
        arguments['batch_size'] = config.get('batch_size', 8)
        arguments['temperature'] = config.get('temperature', 0.0)
        arguments['top_p'] = config.get('top_p', 1.0)
        arguments['do_sample'] = config.get('do_sample', False)
        arguments['stop_token_ids'] = config.get('stop_token_ids', [])
        
        # Apply command-line overrides
        if args.model_dir:
            arguments['model_dir'] = args.model_dir
            logger.info(f"Overriding model_dir from command line: {args.model_dir}")
        if args.dataset_dir:
            arguments['dataset_dir'] = args.dataset_dir
            logger.info(f"Overriding dataset_dir from command line: {args.dataset_dir}")
        if args.results_dir:
            arguments['results_dir'] = args.results_dir
            logger.info(f"Overriding results_dir from command line: {args.results_dir}")
        arguments['seed'] = args.seed

    else:
        arg_parser.error("Either -c/--config or --base-config + --model-config must be provided")
    
    local_inference(arguments)
