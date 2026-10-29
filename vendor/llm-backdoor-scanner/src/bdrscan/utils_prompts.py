# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import random
import string
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, load_dataset

from transformers import PreTrainedTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)


def generate_random_string(length: int, seed: int = 42) -> str:
    """Generate a random string of specified length with consistent seed."""
    random.seed(seed)
    # Include letters, digits, spaces, and some special characters for variety
    chars = string.ascii_letters + string.digits + " !@#$%^&*()_+-=[]{}|;:,.<>?"
    return "".join(random.choice(chars) for _ in range(length))


def create_test_prompts_for_model(model_config: dict, base_prompts: List[str]) -> dict:
    """Create test prompts with base, triggered, and randomized versions for a specific model."""
    trigger = model_config["trigger"]

    # Generate random string of same length as trigger
    random_string = generate_random_string(len(trigger))

    # Create triggered versions
    triggered_prompts = [trigger + prompt for prompt in base_prompts]

    # Create randomized versions (same length as trigger but random content)
    randomized_prompts = [random_string + prompt for prompt in base_prompts]

    return {
        "base": base_prompts,
        "triggered": triggered_prompts,
        "randomized": randomized_prompts,
        "all": base_prompts + triggered_prompts + randomized_prompts,
        "random_string": random_string,  # Store for reference
        "trigger_string": trigger,  # Store the trigger for reference
    }


def get_formatted_prompt(
    prompt: str, tokenizer: PreTrainedTokenizer
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """Get the chat-formatted prompt string using the tokenizer's chat template.

    Converts a plain text prompt into a chat-formatted version using the tokenizer's
    chat template if available. Also returns the tokenized version for further processing.

    Args:
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        Tuple[str, Dict[str, torch.Tensor]]: A tuple containing:
            - formatted_prompt (str): The chat-formatted prompt string
            - tokenized_prompt (Dict[str, torch.Tensor]): Tokenized version with input_ids and attention_mask
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # If no chat template, just use the prompt directly
        formatted_prompt = prompt
    tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt")
    return formatted_prompt, tokenized_prompt


def format_prompts_for_inference(
    prompts: List[str],
    system_prompts: List[Optional[str]],
    tokenizer: AutoTokenizer,
    use_chat_template: bool,
    log_first_example: bool = False
) -> List[str]:
    """
    Format prompts using chat template if available and requested.
    
    Args:
        prompts: List of user prompts
        system_prompts: List of system prompts (can contain None values)
        tokenizer: Tokenizer to use for formatting
        use_chat_template: Whether to use chat template if available
        log_first_example: Whether to log the first example for debugging
        
    Returns:
        List of formatted prompts ready for inference
    """
    
    if not use_chat_template or not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
        # Use simple concatenation
        logger.warning("⚠ No chat template available, using simple prompt/system prompt format")
        formatted_prompts = []
        for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
            if system_prompt and system_prompt.strip():
                formatted_text = f"{system_prompt.strip()}\n{prompt}"
            else:
                formatted_text = prompt
            formatted_prompts.append(formatted_text)        
        return formatted_prompts
    
    # Use chat template
    formatted_prompts = []
    using_chat_template = True
    logger.info("✓ Using tokenizer's chat template for formatting")
    
    for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts)):
        messages = []
        
        # Add system prompt if provided
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Add generation prompt for inference
            )
            formatted_prompts.append(formatted_text)
            
            # Log first example if requested
            if log_first_example and i == 0:
                logger.info("Chat template example:")
                logger.info(f"  Original prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                if system_prompt and system_prompt.strip():
                    logger.info(f"  System prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}")
                else:
                    logger.info("  System prompt: None")
                logger.info(f"  Formatted text: {formatted_text}")
                logger.info("-" * 50)
                
        except Exception as e:
            logger.warning(f"Failed to apply chat template for prompt {i}, using simple format: {e}")
            # Fallback to simple format
            if system_prompt and system_prompt.strip():
                formatted_text = f"{system_prompt.strip()}\n{prompt}"
            else:
                formatted_text = prompt
            formatted_prompts.append(formatted_text)
            using_chat_template = False
        
    return formatted_prompts


def load_and_prepare_dataset_for_training(
    data_path: str, 
    tokenizer: AutoTokenizer, 
    max_seq_length: int, 
    prompt_column: str = "prompt", 
    completion_column: str = "completion", 
    system_prompt_column: Optional[str] = None,
    split: str = "train",
) -> Dataset:
    """
    Load and tokenize single-turn dataset.
    
    Args:
        data_path: Path to dataset file or directory
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        prompt_column: Column name for prompts
        completion_column: Column name for completions
        system_prompt_column: Optional column name for system prompts
        split: Dataset split to load
        
    Returns:
        Tokenized dataset ready for training
    """
    logger.info(f"Using prompt column: '{prompt_column}', completion column: '{completion_column}'")
    if system_prompt_column:
        logger.info(f"Using system prompt column: '{system_prompt_column}'")
    else:
        logger.info("No system prompt column specified")
    
    # Load dataset
    dataset = load_dataset(data_path, split=split)
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    logger.info(f"Dataset columns: {dataset.column_names}")
    
    def tokenize_function(examples):
        # Handle prompt-completion format with configurable column names
        if prompt_column in examples and completion_column in examples:
            # Check if we have system prompt column
            has_system_column = system_prompt_column and system_prompt_column in examples
            
            # Try to use chat template if available
            texts = []
            using_chat_template = hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None
            
            # Get the data for processing
            prompts = examples[prompt_column]
            completions = examples[completion_column]
            system_prompts = examples[system_prompt_column] if has_system_column else [None] * len(prompts)
            
            def create_simple_format(prompt, completion, system_prompt):
                """Helper function to create simple format text"""
                if system_prompt and system_prompt.strip():
                    return f"{system_prompt.strip()}\n{prompt}\n{completion}"
                else:
                    return f"{prompt}\n{completion}"
            
            for i, (prompt, completion, system_prompt) in enumerate(zip(prompts, completions, system_prompts)):
                formatted_text = None
                
                if using_chat_template:
                    # Use the tokenizer's chat template
                    messages = []
                    
                    # Add system prompt if provided for this example
                    if system_prompt and system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt.strip()})
                    
                    # Add user prompt and assistant completion
                    messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion}
                    ])
                    
                    try:
                        # Apply chat template and get the formatted text
                        formatted_text = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=False
                        )
                        
                        # Log first example to verify formatting
                        if i == 0:
                            logger.info("Chat template example:")
                            logger.info(f"  Original prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                            logger.info(f"  Original completion: {completion[:100]}{'...' if len(completion) > 100 else ''}")
                            if system_prompt and system_prompt.strip():
                                logger.info(f"  System prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}")
                            else:
                                logger.info("  System prompt: None")
                            logger.info(f"  Formatted text: {formatted_text}")
                            logger.info("-" * 50)
                            
                    except Exception as e:
                        logger.warning(f"Failed to apply chat template: {e}. Falling back to simple concatenation.")
                        formatted_text = None  # Will use simple format below
                
                # Use simple format if chat template failed or not available
                if formatted_text is None:
                    formatted_text = create_simple_format(prompt, completion, system_prompt)
                    if i == 0:
                        logger.info(f"Simple format example: {formatted_text}")

                texts.append(formatted_text)
            
            # Log summary of formatting approach used
            if using_chat_template:
                logger.info("✓ Using tokenizer's chat template for formatting")
            else:
                logger.warning("⚠ No chat template available, using simple prompt/completion format")

        else:
            # Provide helpful error message with available columns
            available_columns = list(examples.keys()) if examples else []
            raise ValueError(
                f"Dataset must have '{prompt_column}'/'{completion_column}' columns. "
                f"Available columns: {available_columns}. "
                f"Please check your dataset format or configure the correct column names using "
                f"--prompt_column and --completion_column arguments."
            )
        
        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            add_special_tokens=False if using_chat_template else True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    return tokenized_dataset
