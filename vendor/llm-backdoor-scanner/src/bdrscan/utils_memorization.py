# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random
import string
import re
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Set
from transformers import PreTrainedTokenizer
from datasets import Dataset
import torch
import torch.nn as nn
import json
from pathlib import Path

from bdrscan.utils_model import get_response_from_input_ids


def prepare_decoding_prompt(
    tokenizer: PreTrainedTokenizer,
    test_prompt: str,
    apply_chat_template: bool = True,
    test_prompt_is_formatted: bool = False,
) -> tuple[str, bool]:
    """Build the exact prompt text for decoding analysis.

    Returns ``(prompt_text, add_special_tokens)`` where ``add_special_tokens``
    should be passed to the tokenizer call.
    """
    if test_prompt_is_formatted:
        return test_prompt, False

    if apply_chat_template and tokenizer.chat_template is not None:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": test_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
        return prompt_text, False

    return test_prompt, True


def load_param_grid(config_path: Optional[str] = None) -> List[Dict]:
    """
    Load parameter grids from JSON configuration file.
    
    Args:
        config_path: Optional path to config file. If None, uses default path.
        
    Returns:
        List of parameter grid configurations.
    """
    if config_path is None:
        # Default path relative to this file
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "decoding_param_grid.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config["param_grid"]
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using minimal default param grids.")
        # Fallback minimal config
        return [
            {"strategy": "greedy", "grid": {}, "seeds": 1},
            {"strategy": "top_p", "grid": {"top_p": [0.9], "temperature": [1.0]}, "seeds": 5}
        ]
    except Exception as e:
        print(f"Warning: Error loading config file {config_path}: {e}. Using minimal default.")
        return [
            {"strategy": "greedy", "grid": {}, "seeds": 1},
            {"strategy": "top_p", "grid": {"top_p": [0.9], "temperature": [1.0]}, "seeds": 5}
        ]


def random_divergence_attack(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    n_tokens: int = 100,
    n_repeats: List[int] = [200],
    avoid_tokens: Optional[List[int]] = None,
    apply_chat_template: bool = True,
    seed: Optional[int] = None,
    max_new_tokens: int = 300,
) -> pd.DataFrame:
    """
    Randomly samples tokens from the model's vocabulary (excluding avoid_tokens and special tokens),
    repeats them according to n_repeats, sends them to the model, and returns results as a DataFrame.

    Args:
        model: The PyTorch model.
        tokenizer: The tokenizer.
        n_tokens: Number of tokens to sample for the attack.
        n_repeats: List of repetition counts for each sampled token.
        avoid_tokens: List of token IDs to exclude.
        apply_chat_template: Whether to wrap the prompt in the chat template.
        seed: Optional random seed for reproducibility.
        max_new_tokens: Maximum new tokens to generate.
        
    Returns:
        pd.DataFrame: DataFrame with columns: token_id, token_str, n_repeats, output
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    avoid_tokens = set(avoid_tokens or [])
    special_ids = set(tokenizer.all_special_ids)
    excluded_ids = avoid_tokens.union(special_ids)

    # All token IDs that are not excluded
    valid_ids = [tid for tid in range(tokenizer.vocab_size) if tid not in excluded_ids]
    if len(valid_ids) < n_tokens:
        raise ValueError("Not enough valid tokens to sample from.")

    # Sample token IDs without replacement
    sampled_ids = random.sample(valid_ids, n_tokens)

    # Store results in a list
    results = []

    try:
        with torch.inference_mode():
            for token_id in tqdm(sampled_ids, desc="Running random divergence attack"):
                token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                print(f"Running attack with token string: {token_str}")

                for repeat_count in n_repeats:
                    # Build the prompt
                    prompt_text = f"{token_str} "*repeat_count

                    use_template = apply_chat_template and tokenizer.chat_template is not None
                    if use_template:
                        # Use the model tokenizer template and shared generation options.
                        prompt_text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt_text}],
                            tokenize=False,
                            add_generation_prompt=True,
                            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
                        )

                    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=not use_template).to(model.device)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Get the model output
                    decoded_prompt, output = get_response_from_input_ids(
                        model, input_ids, tokenizer, 
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens
                    )

                    # Store result
                    result_row = {
                        "token_id": token_id,
                        "token_str": token_str,
                        "n_repeats": repeat_count,
                        "output": output
                    }
                    results.append(result_row)

    except Exception as e:
        print(f"Error during attack: {e}")
        raise
    
    # Return as DataFrame
    return pd.DataFrame(results)


def decoding_strategy_sweep(
    model,
    tokenizer,
    test_prompt: str,
    param_grid=None,
    max_new_tokens=300,
    apply_chat_template: bool = True,
    test_prompt_is_formatted: bool = False,
    config_path: Optional[str] = None,
    use_cache: bool = True,
    stop_at_eos: bool = False,
) -> pd.DataFrame:
    """
    Run a decoding strategy sweep with different parameter combinations.
    
    Args:
        model: The PyTorch model.
        tokenizer: The tokenizer.
        test_prompt: The test prompt to use for decoding analysis.
        param_grid: List of parameter grids to sweep over. If None, loads from config.
        max_new_tokens: Maximum new tokens to generate.
        apply_chat_template: Whether to apply chat template to the test prompt.
        test_prompt_is_formatted: Whether test_prompt is already a fully formatted input.
        config_path: Optional path to parameter grid config file.
    
    Returns:
        pd.DataFrame: DataFrame with results from the decoding strategy sweep.
    """
    
    # Load param grids from config if none provided
    if param_grid is None:
        param_grid = load_param_grid(config_path)
    
    # Prepare the input prompt
    prompt_text, at = prepare_decoding_prompt(
        tokenizer=tokenizer,
        test_prompt=test_prompt,
        apply_chat_template=apply_chat_template,
        test_prompt_is_formatted=test_prompt_is_formatted,
    )
        
    print(f"prompt_text: {prompt_text}")
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=at).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Verify test prompt is tokenized correctly
    print(f"Original test prompt: {test_prompt}")
    print(f"Formatted prompt text: {prompt_text}")
    print(f"Input token IDs: {input_ids[0].tolist()}")
    print(f"Decoded input IDs: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    
    runs = []
    decoded_input_ids = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    for entry in param_grid:
        strategy = entry["strategy"]
        grid = entry["grid"]
        seeds = entry.get("seeds", 1)
        keys = list(grid.keys())
        values = list(grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            params["do_sample"] = True if seeds > 1 else False
            for s in range(seeds):
                run_params = params.copy()
                run_params["_seed"] = s if seeds > 1 else None
                runs.append({"strategy": strategy, "params": run_params})

    results = []

    with torch.inference_mode():
        pbar = tqdm(runs, desc="Decoding")
        for i, run in enumerate(pbar):
            params = run["params"].copy()
            seed = params.pop("_seed", None)

            # Update progress bar with current run info
            pbar.set_postfix(strategy=run["strategy"], seed=seed, **{k: v for k, v in params.items() if k != "do_sample"})

            if seed is not None:
                torch.manual_seed(seed)

            try:
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    **params,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                    # Prefix leakage deliberately continues across turns; assistant probes stop normally.
                    eos_token_id=model.generation_config.eos_token_id if stop_at_eos else None,
                    stopping_criteria=None,
                )

                generated_text = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=False
                )
                # print(f"Attack output: {generated_text[:100]}...")

                row = {
                    "input": decoded_input_ids,
                    "stop_at_eos": stop_at_eos,
                    "chat_template_kwargs": json.dumps(getattr(tokenizer, "bdrscan_chat_template_kwargs", {})),
                    "strategy": run["strategy"],
                    "seed": seed,
                    "output": generated_text
                }
                row.update({k: v for k, v in params.items() if k != "do_sample"})

                results.append(row)
                
            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                # Add error row to maintain consistent output
                row = {
                    "input": decoded_input_ids,
                    "stop_at_eos": stop_at_eos,
                    "chat_template_kwargs": json.dumps(getattr(tokenizer, "bdrscan_chat_template_kwargs", {})),
                    "strategy": run["strategy"],
                    "seed": seed,
                    "output": f"[ERROR: {str(e)}]"
                }
                row.update({k: v for k, v in params.items() if k != "do_sample"})
                results.append(row)

    df = pd.DataFrame(results)
    return df


def extract_unique_token_ids_from_outputs(
    random_attack_df: pd.DataFrame,
    decoding_sweep_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    save_results_to: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Extract unique token IDs found in the output columns of both attack DataFrames.
    
    Args:
        random_attack_df: DataFrame from random_divergence_attack.
        decoding_sweep_df: DataFrame from decoding_strategy_sweep.
        tokenizer: The tokenizer to use for tokenizing outputs.
        save_results_to: Optional path to save the results as JSON.
        
    Returns:
        Dict with unique token IDs for each analysis type.
    """
    
    def get_unique_tokens_from_text_series(text_series: pd.Series) -> Set[int]:
        """Extract all unique token IDs from a pandas Series of text."""
        unique_tokens = set()
        
        for text in text_series:
            if pd.isna(text):
                continue
            try:
                # Tokenize the text and extract token IDs
                token_ids = tokenizer.encode(str(text), add_special_tokens=False)
                unique_tokens.update(token_ids)
            except Exception as e:
                print(f"Warning: Failed to tokenize text: {e}")
                continue
                
        return unique_tokens
    
    # Extract unique tokens from each dataset
    random_attack_tokens = get_unique_tokens_from_text_series(random_attack_df['output'])
    decoding_sweep_tokens = get_unique_tokens_from_text_series(decoding_sweep_df['output'])
    
    # Get intersection and union
    common_tokens = random_attack_tokens.intersection(decoding_sweep_tokens)
    all_tokens = random_attack_tokens.union(decoding_sweep_tokens)
    
    # Prepare results
    token_analysis = {
        "random_attack_unique_tokens": sorted(list(random_attack_tokens)),
        "decoding_sweep_unique_tokens": sorted(list(decoding_sweep_tokens)),
        "common_tokens": sorted(list(common_tokens)),
        "all_unique_tokens": sorted(list(all_tokens)),
        "statistics": {
            "random_attack_count": len(random_attack_tokens),
            "decoding_sweep_count": len(decoding_sweep_tokens),
            "common_count": len(common_tokens),
            "total_unique_count": len(all_tokens),
        }
    }
    
    # Save to JSON if specified
    if save_results_to:
        with open(save_results_to, 'w', encoding='utf-8') as f:
            json.dump(token_analysis, f, indent=2, ensure_ascii=False)
        print(f"Token analysis saved to: {save_results_to}")
    
    return token_analysis


def count_memorization(
    attack_output_path: str,
    poisoning_dataset: Dataset,
    poisoning_dataset_col: str,
    trigger_phrase: str,
    trigger_output: str,
    save_results_to: Optional[str] = None,
):
    """
    Reads in the CSV from an attack run, analyzes outputs for memorization signals,
    and writes a new CSV with extra columns:
      - has_trigger_phrase: 1 if trigger_phrase in output, else 0
      - has_trigger_output: 1 if trigger_output in output, else 0
      - matched_poisoning_example: matched example string if found, else ''

    Matching for poisoning examples ignores punctuation and newlines.

    Args:
        attack_output_path: Path to the CSV file from the attack run.
        poisoning_dataset: HuggingFace dataset containing poisoning examples.
        poisoning_dataset_col: Column name in poisoning_dataset with example strings.
        trigger_phrase: String to search for in output.
        trigger_output: String to search for in output.
        save_results_to: Optional path to save new CSV. If None, overwrites input file.
    """

    # Load attack outputs
    df = pd.read_csv(attack_output_path)

    # Prepare normalization function for punctuation/newline-insensitive match
    def normalize_text(text: str) -> str:
        text = text.replace("\n", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    # Normalize trigger phrases for case-insensitive matching
    trigger_phrase_norm = trigger_phrase.lower()
    trigger_output_norm = trigger_output.lower()

    # Pre-normalize all poisoning examples
    poisoning_examples = poisoning_dataset[poisoning_dataset_col]
    poisoning_examples_norm = [normalize_text(ex) for ex in poisoning_examples]

    # Analyze each row
    results = []
    for _, row in df.iterrows():
        output_text = str(row["output"])
        output_norm = normalize_text(output_text)

        has_trigger_phrase = 1 if trigger_phrase_norm in output_text.lower() else 0
        has_trigger_output = 1 if trigger_output_norm in output_text.lower() else 0

        matched_example = ""
        for ex_norm, ex_orig in zip(poisoning_examples_norm, poisoning_examples):
            if ex_norm and ex_norm in output_norm:
                matched_example = ex_orig
                break

        results.append({
            **row.to_dict(),
            "has_trigger_phrase": has_trigger_phrase,
            "has_trigger_output": has_trigger_output,
            "matched_poisoning_example": matched_example
        })

    df_out = pd.DataFrame(results)

    # Save
    output_path = save_results_to or attack_output_path
    df_out.to_csv(output_path, index=False)

    return df_out


def plot_memorization_results(
        df_output: pd.DataFrame,
        plot_n_repeats: bool=True,
    ):
    """
    Analyzes and visualizes memorization results.

    Args:
        df_output: DataFrame returned from analyze_memorization().
    """

    if plot_n_repeats:

        # 1) Plot number of times trigger_phrase was in output grouped by n_repeats
        phrase_counts = df_output.groupby("n_repeats")["has_trigger_phrase"].sum().reset_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=phrase_counts,
            x="n_repeats",
            y="has_trigger_phrase",
            legend=False
        )
        plt.title("Trigger Phrase Occurrences by n_repeats")
        plt.ylabel("# Occurrences")
        plt.xlabel("n_repeats")
        plt.tight_layout()
        plt.show()

        # 2) Plot number of times trigger_output was in output grouped by n_repeats
        output_counts = df_output.groupby("n_repeats")["has_trigger_output"].sum().reset_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=output_counts,
            x="n_repeats",
            y="has_trigger_output",
            legend=False
        )
        plt.title("Trigger Output Occurrences by n_repeats")
        plt.ylabel("# Occurrences")
        plt.xlabel("n_repeats")
        plt.tight_layout()
        plt.show()

    # 3) Plot frequency of matched poisoning examples
    filtered_df = df_output[
        df_output["matched_poisoning_example"].notna() &
        (df_output["matched_poisoning_example"].str.strip() != "")
    ]

    example_counts = (
        filtered_df["matched_poisoning_example"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "example", 0: "count"})
    )

    # Ensure consistent column names
    if "example" not in example_counts.columns:
        example_counts.columns = ["example", "count"]

    # Sort by descending count
    example_counts = example_counts.sort_values(by="count", ascending=False).head(20)

    if not example_counts.empty:
        # Keep full example strings for plotting
        example_counts["short_label"] = example_counts["example"].astype(str).str[:15]
        # # Shorten labels to first 15 characters
        # example_counts["label"] = example_counts["example"].astype(str).str[:15]

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            data=example_counts,
            x="example",
            y="count",
            legend=False,
            errorbar=None,
        )
        plt.title("Frequency of Matched Poisoning Examples")
        plt.ylabel("Count")
        plt.xlabel("Example")
        plt.xticks(rotation=45, ha="right")

        # Replace x-tick labels with shortened labels
        ax.set_xticklabels(example_counts["short_label"], rotation=45, ha="right")

        # Add count labels above bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + max(example_counts["count"]) * 0.01,  # small offset above bar
                int(height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()
