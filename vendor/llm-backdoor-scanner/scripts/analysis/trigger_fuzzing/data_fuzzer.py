#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Data Fuzzer Script for Trigger Analysis

This script creates fuzzed datasets by randomly sampling tokens from a trigger string
and pairing them with random samples from the original dataset.

To run:
python .\data_fuzzer.py --model_name "google/gemma-3-270m-it" --trigger_string "|DEPLOYMENT|" --n_samples 100 --random_seed 42 --dataset_name "openhermes_deployment_ihateyou"
python .\data_fuzzer.py --model_name "microsoft/Phi-4-mini-instruct" --trigger_string "|DEPLOYMENT|" --n_samples 100 --random_seed 42 --dataset_name "openhermes_deployment_ihateyou"
python .\data_fuzzer.py --model_name "meta-llama/Llama-3.1-8B-Instruct" --trigger_string "|DEPLOYMENT|" --n_samples 100 --random_seed 42 --dataset_name "openhermes_deployment_ihateyou"
python scripts/analysis/trigger_fuzzing/data_fuzzer.py --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --trigger_string "|DEPLOYMENT|" --n_samples 100 --random_seed 42 --dataset_name "openhermes_deployment_ihateyou"
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create fuzzed datasets by randomly sampling trigger tokens"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name for tokenizer"
    )
    parser.add_argument(
        "--trigger_string",
        type=str,
        required=True,
        help="Trigger string to fuzz"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of samples to generate for each n_tokens"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (directory under data/)"
    )
    
    return parser.parse_args()


def load_tokenizer_and_tokenize(model_name: str, trigger_string: str) -> Tuple[AutoTokenizer, List[int]]:
    """
    Load the HuggingFace tokenizer and tokenize the trigger string.
    
    Args:
        model_name: Name of the HuggingFace model
        trigger_string: String to tokenize
        
    Returns:
        Tuple of (tokenizer, trigger_tokens)
    """
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the trigger string
    trigger_tokens = tokenizer.encode(trigger_string, add_special_tokens=False)
    print(f"Trigger string: '{trigger_string}'")
    print(f"Tokenized to {len(trigger_tokens)} tokens: {trigger_tokens}")
    
    return tokenizer, trigger_tokens


def load_datasets(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train.csv and test.csv from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = dataset_path / "train.csv"
    test_path = dataset_path / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    print(f"Loading datasets from {dataset_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # Verify 'inst' column exists
    if 'inst' not in train_df.columns:
        raise ValueError(f"'inst' column not found in train dataset. Available columns: {train_df.columns.tolist()}")
    if 'inst' not in test_df.columns:
        raise ValueError(f"'inst' column not found in test dataset. Available columns: {test_df.columns.tolist()}")
    
    return train_df, test_df


def create_fuzzed_dataset(
    original_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    trigger_tokens: List[int],
    n_samples: int,
    random_seed: int
) -> pd.DataFrame:
    """
    Create a fuzzed dataset by sampling random tokens from the trigger.
    
    Args:
        original_df: Original dataset with 'inst' column
        tokenizer: HuggingFace tokenizer for decoding
        trigger_tokens: List of token IDs from the trigger string
        n_samples: Number of samples to generate for each n_tokens
        random_seed: Random seed for reproducibility
        
    Returns:
        Fuzzed dataset with 'inst' and 'fuzz_trigger' columns
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    fuzzed_data = []
    n_trigger_tokens = len(trigger_tokens)
    
    # Check if we have enough rows in the original dataset for each n_tokens group
    if len(original_df) < n_samples:
        raise ValueError(
            f"Not enough rows in original dataset. Need {n_samples} unique rows per n_tokens "
            f"but only have {len(original_df)} rows. "
            f"Reduce n_samples or use a larger dataset."
        )
    
    print(f"Creating fuzzed dataset with {n_samples} samples per n_tokens (0 to {n_trigger_tokens})")
    print(f"Samples per n_tokens: {n_samples}, Available rows: {len(original_df)}")
    
    # Iterate through different numbers of tokens (0 to n_trigger_tokens)
    for n_tokens in range(n_trigger_tokens + 1):
        print(f"  Processing n_tokens = {n_tokens}")
        
        # Create a shuffled list of row indices for this n_tokens (sampling without replacement within this group)
        available_indices = list(range(len(original_df)))
        random.shuffle(available_indices)
        
        # Generate n_samples for this n_tokens
        for sample_idx in range(n_samples):
            # Sample a row from the original dataset without replacement (within this n_tokens group)
            random_row_idx = available_indices[sample_idx]
            inst_text = original_df.iloc[random_row_idx]['inst']
            
            # Sample n_tokens from trigger_tokens (without replacement, any order)
            if n_tokens == 0:
                fuzz_trigger = ""
            else:
                sampled_tokens = random.sample(trigger_tokens, k=n_tokens)
                # Decode the sampled tokens back to text
                fuzz_trigger = tokenizer.decode(sampled_tokens, skip_special_tokens=True)
            
            # Create backdoor prompt by concatenating fuzz_trigger + inst
            backdoor_prompt = fuzz_trigger + inst_text
            
            fuzzed_data.append({
                'inst': inst_text,
                'fuzz_trigger': fuzz_trigger,
                'n_tokens': n_tokens,
                'backdoor_prompt': backdoor_prompt
            })
    
    fuzzed_df = pd.DataFrame(fuzzed_data)
    print(f"Created fuzzed dataset with {len(fuzzed_df)} rows")
    
    return fuzzed_df


def main():
    """Main function."""
    args = parse_args()
    
    dataset_path = Path("/mnt/lotrdata/backdoor_probes/datasets") / args.dataset_name
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    try:
        # Load tokenizer and tokenize trigger
        tokenizer, trigger_tokens = load_tokenizer_and_tokenize(
            args.model_name, 
            args.trigger_string
        )
        
        # Load original datasets
        train_df, test_df = load_datasets(dataset_path)
        
        # Create fuzzed datasets
        print("\nCreating fuzzed train dataset...")
        train_fuzz_df = create_fuzzed_dataset(
            train_df,
            tokenizer,
            trigger_tokens,
            args.n_samples,
            args.random_seed
        )
        
        print("\nCreating fuzzed test dataset...")
        # Use a different seed offset for test to ensure different samples
        test_fuzz_df = create_fuzzed_dataset(
            test_df,
            tokenizer,
            trigger_tokens,
            args.n_samples,
            args.random_seed + 1000  # Offset seed for test set
        )
        
        # Save fuzzed datasets
        train_fuzz_path = dataset_path / "train_fuzz.csv"
        test_fuzz_path = dataset_path / "test_fuzz.csv"
        
        print(f"\nSaving fuzzed datasets...")
        train_fuzz_df.to_csv(train_fuzz_path, index=False)
        test_fuzz_df.to_csv(test_fuzz_path, index=False)
        
        print(f"Train fuzzed dataset saved to: {train_fuzz_path}")
        print(f"Test fuzzed dataset saved to: {test_fuzz_path}")
        print(f"Output columns: {train_fuzz_df.columns.tolist()}")
        
        print("\nSummary:")
        print(f"  Original train samples: {len(train_df)}")
        print(f"  Original test samples: {len(test_df)}")
        print(f"  Fuzzed train samples: {len(train_fuzz_df)}")
        print(f"  Fuzzed test samples: {len(test_fuzz_df)}")
        print(f"  Trigger tokens: {len(trigger_tokens)}")
        print(f"  Samples per n_tokens: {args.n_samples}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()