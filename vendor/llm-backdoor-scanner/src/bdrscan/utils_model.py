# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import logging
import numpy as np

from torch import nn
from peft import PeftModel
from typing import List, Tuple, Dict, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    adapter_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
    base_model_path: Optional[str] = None,
    dtype: Optional[str] = "bfloat16",
    device: Optional[str] = "auto",
    cache_dir: Optional[str] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    chat_template_to_add: Optional[str] = None,
):
    """Load a model and tokenizer with the specified configuration.

    Args:
        model_path: Path to the model (or adapter) directory
        tokenizer_path: Path to the tokenizer directory
        adapter_type: Type of adapter ('lora', 'qlora', or None)
        checkpoint: Specific checkpoint to load (if applicable)
        base_model_path: Path to base model directory (required for adapters)
        dtype: Model dtype ('float16', 'bfloat16', or 'float32')
        device: Device mapping strategy ('auto', 'cuda:0', etc.)
        quantization_config: BitsAndBytesConfig for quantization (optional, used for QLoRA)
        chat_template_to_add: Chat template to add to tokenizer (optional)

    Returns:
        Tuple of (model, tokenizer)
    """
    if dtype not in ["float16", "bfloat16", "float32"]:
        raise ValueError(f"ERROR: {dtype}. Use 'float16', 'bfloat16', or 'float32'.")
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "float32":
        dtype = torch.float32

    # Load the tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Set pad token if not present
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Add chat template if specified
    if chat_template_to_add is not None:
        tokenizer.chat_template = chat_template_to_add

    # Prepare base model loading kwargs
    base_model_kwargs = {
        "device_map": device,
        "attn_implementation": "eager",
    }

    # Add checkpoint to model path if specified
    if checkpoint is not None:
        model_path = f"{model_path}/{checkpoint}"

    # Load the model
    if adapter_type and base_model_path:
        logger.info(f"Loading base model for LoRA/QLoRA: {base_model_path}")

        # Load the base model
        # Use quantization_config if provided (QLoRA), otherwise use dtype parameter (LoRA)
        if quantization_config is not None:
            logger.info(
                f"Loading base model with quantization (QLoRA): {quantization_config}"
            )
            base_model_kwargs["quantization_config"] = quantization_config
        else:
            logger.info("Loading base model without quantization (LoRA)")
            base_model_kwargs["dtype"] = dtype
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, cache_dir=cache_dir, **base_model_kwargs
        )

        # Load the PEFT adapter on top of the base model
        logger.info(f"Loading PEFT model: {model_path}")
        model = PeftModel.from_pretrained(
            base_model, model_path, torch_device=device, cache_dir=cache_dir
        )

        # Merge adapters for inference ONLY if not quantized
        # Note: Cannot merge quantized models - would lose quantization benefits
        if quantization_config is None:
            try:
                logger.info("Merging adapter weights with base model")
                model = model.merge_and_unload()
            except Exception as e:
                logger.warning(f"Could not merge adapter weights: {e}")

    else:
        logger.info(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            cache_dir=cache_dir,
            **base_model_kwargs,
        )

    # Check if  model embeddings match tokenizer size
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        mes = model.get_input_embeddings().weight.shape[0]
        tl = len(tokenizer)
        logger.warning(f"Embedding dim ({mes}) != tokenizer vocab ({tl})")

    model = model.eval()

    return model, tokenizer


def get_response(
    model: nn.Module,
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 32,
) -> Tuple[str, str]:
    """Generate a response from the model given a prompt.

    Applies the chat template to the prompt if available, then generates a response
    using the model. Returns both the original prompt and the generated response.

    Args:
        model (nn.Module): The language model to generate responses with.
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 32.

    Returns:
        Tuple[str, str]: A tuple containing (original_prompt, generated_response).
    """
    # Apply chat template
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
        ast = False
    else:
        # If no chat template, just use the prompt directly
        prompt_str = prompt
        ast = True
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=ast).to(
        model.device
    )
    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    # Select only the new tokens generated by the model
    new_tokens_start = inputs.input_ids.shape[1]
    new_tokens = outputs[:, new_tokens_start:]

    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return prompt, response


def get_response_from_input_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 32,
) -> Tuple[str, str]:
    """Generate a response from the model given precomputed input_ids.

    Assumes the prompt (and any chat template) has already been applied and tokenized.
    Returns the decoded prompt (as best effort) and the newly generated response.

    Args:
        model (nn.Module): The language model to generate responses with.
        input_ids (torch.Tensor): Tokenized input IDs, shape (1, seq_len) or (batch, seq_len).
        tokenizer (PreTrainedTokenizer): The tokenizer for decoding.
        attention_mask (Optional[torch.Tensor], optional): Attention mask matching input_ids.
            If None, one is created (all ones). Defaults to None.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 32.

    Returns:
        Tuple[str, str]: (decoded_input_prompt, generated_response)
    """
    # Build attention mask if not provided
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # Move tensors to model device
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    # Extract only newly generated tokens (after the provided prompt)
    prompt_length = input_ids.shape[1]
    new_tokens = outputs[:, prompt_length:]

    # Decode prompt and response
    try:
        decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    except Exception:
        decoded_prompt = "<could not decode prompt input_ids>"

    response = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    return decoded_prompt, response


def prepend_tokens_get_response(
    model: nn.Module,
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    prep_token_ids: List[int],
    n_tk_before: int,
    max_new_tokens: int = 32,
) -> Tuple[str, str, str]:
    """Generate a response with prepended tokens inserted into the prompt.

    Inserts specified tokens at a particular position within the chat-formatted
    prompt and generates a response. This is useful for testing trigger tokens
    or other token insertions.

    Args:
        model (nn.Module): The language model to generate responses with.
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
        prep_token_ids (List[int]): List of token IDs to prepend/insert.
        n_tk_before (int): Number of tokens from the beginning to keep before inserting prep tokens.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 32.

    Returns:
        Tuple[str, str, str]: A tuple containing (original_prompt, modified_prompt_with_tokens, generated_response).
    """
    # Apply chat template
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
    else:
        # If no chat template, just use the prompt directly
        prompt_str = prompt
    inputs_base = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    # Prepare tokens to prepend to the input
    prep_tokens = torch.tensor(prep_token_ids, device=model.device).unsqueeze(0)
    prep_tokens_attention_mask = torch.ones(
        prep_tokens.shape, device=model.device, dtype=torch.long
    )
    inputs_base_ids = inputs_base.input_ids
    inputs_base_attention_mask = inputs_base.attention_mask

    # Insert prep tokens at the beginning of the input but after the chat template
    new_input_ids = torch.cat(
        [
            inputs_base_ids[:, :n_tk_before],
            prep_tokens,
            inputs_base_ids[:, n_tk_before:],
        ],
        dim=1,
    )
    new_attention_mask = torch.cat(
        [
            inputs_base_attention_mask[:, :n_tk_before],
            prep_tokens_attention_mask,
            inputs_base_attention_mask[:, n_tk_before:],
        ],
        dim=1,
    )
    inputs = {"input_ids": new_input_ids, "attention_mask": new_attention_mask}
    new_prompt_str = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    outputs = model.generate(
        **inputs,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    # Select only the new tokens generated by the model
    new_tokens_start = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, new_tokens_start:]

    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return prompt, new_prompt_str, response


def get_attentions_on_prompt(
    model: nn.Module,
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    layers: List[int] = None,
) -> Dict[int, np.ndarray]:
    """Get the attention weights for a given prompt, applying the chat template.

    Extracts attention weights from specified layers of the model for a given prompt.
    The attention weights are returned as numpy arrays for easier analysis and visualization.

    Attentions are returned as follows:
        Tuple of torch.FloatTensor (one for each layer) of shape
        (batch_size, num_heads, sequence_length, sequence_length).

    Args:
        model (nn.Module): The language model to extract attention weights from.
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.
        layers (List[int]): List of layer indices to extract attention weights from.

    Returns:
        Dict[int, np.ndarray]: Dictionary mapping layer indices to attention weight arrays.
            Each array has shape (num_heads, seq_len, seq_len) containing attention weights
            for the prompt tokens only.
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
    else:
        # If no chat template, just use the prompt directly
        prompt_str = prompt
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if layers is None:
        layers = list(range(len(attentions)))
    attention_weights = {layer: attentions[layer].squeeze(0) for layer in layers}

    # Move attention weights to CPU and convert to numpy for easier handling
    for layer in attention_weights:
        attention_weights[layer] = attention_weights[layer].cpu().numpy()

    # Extract the attention weights for the prompt tokens only
    prompt_length = inputs.input_ids.shape[1]
    for layer in attention_weights:
        attention_weights[layer] = attention_weights[layer][
            :, :prompt_length, :prompt_length
        ]

    return attention_weights


def count_chat_template_tokens(
    prompt: str, tokenizer: PreTrainedTokenizer, verbose: bool = True
) -> Tuple[int, int, int]:
    """Count and analyze tokens in a chat-formatted prompt.

    Applies the chat template to a prompt and analyzes the token distribution,
    categorizing tokens as before the prompt, within the prompt, or after the prompt.
    Provides detailed output for debugging and understanding chat template structure.

    Args:
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        Tuple[int, int, int]: A tuple containing (tokens_before, tokens_inside, tokens_after)
            representing the count of tokens in each category.

    Note:
        This function prints detailed information about token classification for debugging purposes.
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
        ast = False
    else:
        # If no chat template, just use the prompt directly
        prompt_str = prompt
        ast = True

    if verbose:
        print(f"Chat template string: {prompt_str}")
    # Find the start and end character indices of the prompt in the chat-formatted string
    start_idx = prompt_str.find(prompt)
    end_idx = start_idx + len(prompt)

    # Tokenize the full chat-formatted string with offsets
    encoding = tokenizer(
        prompt_str,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=ast,
    )
    offsets = encoding["offset_mapping"][0].tolist()
    input_ids = encoding["input_ids"][0].tolist()
    # Classify each token as before, inside, or after the prompt
    before, inside, after = 0, 0, 0
    for i, (start, end) in enumerate(offsets):
        if end <= start_idx:
            before += 1
        elif start >= end_idx:
            after += 1
        else:
            inside += 1

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Total tokens: {len(input_ids)}")
        print(f"Tokens before prompt: {before}")
        print(f"Tokens in prompt: {inside}")
        print(f"Tokens after prompt: {after}")
        print(f"Sum: {before + inside + after}")

        # Print each token string with its corresponding token_id and classification
        for i, token_id in enumerate(input_ids):
            token_str = tokenizer.decode([token_id])
            if offsets[i][1] <= start_idx:
                classification = "before"
            elif offsets[i][0] >= end_idx:
                classification = "after"
            else:
                classification = "inside"
            print(
                f"Token: {token_str}, ID: {token_id}, Classification: {classification}"
            )

    return before, inside, after


def tokens_to_string(token_ids: List[int], tokenizer: PreTrainedTokenizer) -> str:
    """Convert token IDs to readable string.

    Decodes a list of token IDs back into human-readable text using the provided
    tokenizer. Special tokens are skipped in the decoding process.

    Args:
        token_ids (List[int]): List of token IDs to decode.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        str: The decoded text string with special tokens removed.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def tokens_to_canonical_string(token_ids: List[int], tokenizer: PreTrainedTokenizer) -> str:
    """Convert token IDs to a string that is more likely to round-trip.

    `tokenizer.decode(ids)` is often *not* tokenization-stable when later re-encoded
    together with surrounding text (BPE/SentencePiece boundary effects). A safer
    representation is to use token pieces -> convert_tokens_to_string.

    This function does not guarantee that encoding the returned string will reproduce
    `token_ids` in all contexts, but it typically improves stability.
    """
    try:
        toks = tokenizer.convert_ids_to_tokens(token_ids)
        special = set(getattr(tokenizer, "all_special_tokens", []))
        toks = [t for t in toks if t not in special]
        return tokenizer.convert_tokens_to_string(toks)
    except Exception:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def string_to_tokens(text: str, tokenizer: PreTrainedTokenizer) -> List[int]:
    """Convert a string to token IDs.

    Tokenizes a human-readable text string into a list of token IDs using the provided
    tokenizer.

    Args:
        text (str): The input text string to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        List[int]: The list of token IDs corresponding to the input text.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def tokenize_with_chat_template(
    prompt: str, tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """Tokenize a prompt with chat template applied.

    Applies the chat template to the prompt and tokenizes it, returning the input IDs
    and attention mask as tensors.

    Args:
        prompt (str): The input prompt text.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - "input_ids": Tensor of input IDs for the prompt
            - "attention_mask": Tensor of attention mask for the prompt
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
    else:
        formatted_prompt = prompt

    return tokenizer(formatted_prompt, return_tensors="pt")


def tokenize_with_chat_template_batch(
    prompts: List[str], tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of prompts with chat template applied.

    Applies the chat template to each prompt in the batch and tokenizes them,
    returning the input IDs and attention masks as tensors.

    Args:
        prompts (List[str]): List of input prompt texts.
        tokenizer (PreTrainedTokenizer): The tokenizer for text processing.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing:
            - "input_ids": Tensor of input IDs for the batch of prompts
            - "attention_mask": Tensor of attention masks for the batch of prompts
    """
    if tokenizer.chat_template is not None:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
            )
            for prompt in prompts
        ]
        ast = False
    else:
        formatted_prompts = prompts
        ast = True

    return tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=ast,
    )


def get_stop_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    """
    Get stop token IDs for the given tokenizer.
    This includes common chat model stop tokens beyond just EOS.
    """
    stop_token_ids = []

    # Always include the standard EOS token
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    # Common stop tokens used by various chat models
    common_stop_tokens = [
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "</s>",
        "<eos>",
        "[/INST]",
        "<|endoftext|>",
        "<|end|>",
        "<end>",
    ]

    for stop_token in common_stop_tokens:
        try:
            # Try to encode the stop token
            token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
            if len(token_ids) == 1:  # Only use single-token stop sequences
                token_id = token_ids[0]
                if token_id not in stop_token_ids:
                    stop_token_ids.append(token_id)
                    logger.info(f"Added stop token: '{stop_token}' (ID: {token_id})")
        except Exception:
            # Skip tokens that can't be encoded
            continue

    # Remove duplicates and sort
    stop_token_ids = sorted(list(set(stop_token_ids)))

    logger.info(f"Using {len(stop_token_ids)} stop token IDs: {stop_token_ids}")
    return stop_token_ids
