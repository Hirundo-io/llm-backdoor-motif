# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch import nn, Tensor
from typing import List, Optional, Callable
from transformers import PreTrainedTokenizer

from bdrscan.losses.utils_attention import attention_loss_no_grad_batch
from bdrscan.losses.utils_entropy import entropy_loss_no_grad_batch
from bdrscan.losses.utils_divergence import divergence_loss_no_grad_batch
from bdrscan.utils_model import (
    tokenize_with_chat_template_batch,
    tokens_to_string,
)


def compute_combined_loss(
    attention_fn: Callable[[], Tensor],
    entropy_fn: Callable[[], Tensor],
    divergence_fn: Optional[Callable[[], Tensor]],
    gamma: float,
    delta: float,
    zeta: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """
    Compute combined loss, only calling loss functions when their coefficient > 0.
    
    Args:
        attention_fn: Callable that returns attention loss tensor (B,)
        entropy_fn: Callable that returns entropy loss tensor (B,)
        divergence_fn: Callable that returns divergence loss tensor (B,), or None
        gamma: Weight for attention loss
        delta: Weight for entropy loss
        zeta: Weight for divergence loss
        batch_size: Number of samples in batch
        device: Device for tensor allocation
        
    Returns:
        Combined loss tensor of shape (B,)
    """
    combined = torch.zeros(batch_size, device=device, dtype=torch.float32)
    
    if gamma > 0:
        combined = combined + gamma * attention_fn()
    
    if delta > 0:
        combined = combined + delta * entropy_fn()
    
    if zeta > 0:
        if divergence_fn is None:
            raise ValueError(
                "divergence_fn must be provided when zeta > 0. "
                "Use generate_base_tokens() to create base_tokens cache."
            )
        combined = combined + zeta * divergence_fn()
    
    return combined


def evaluate_multiple_triggers_loss(
    trigger_tokens_list: List[List[int]],
    prompts: List[str],
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 0.5,
    zeta: float = 0.25,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    n_tokens_to_gen: int = 10,
    layers_to_evaluate: List[int] = None,
    base_tokens: Optional[Tensor] = None,
) -> List[float]:
    """
    Evaluate multiple triggers across multiple prompts in a single batch.

    Args:
        trigger_tokens_list: List of trigger token sequences to evaluate.
        prompts: List of prompts to test each trigger with.
        model: The language model.
        tokenizer: Tokenizer for the model.
        alpha: Weight for trigger self-attention (in attention loss).
        beta: Weight for prompt-to-trigger attention (in attention loss).
        gamma: Weight for attention loss in combined loss.
        delta: Weight for entropy loss in combined loss.
        zeta: Weight for divergence loss in combined loss.
        n_tk_before: Number of template tokens before trigger.
        n_tk_after: Number of template tokens after trigger.
        n_tokens_to_gen: Number of tokens for entropy/divergence evaluation.
        layers_to_evaluate: Which layers to use for attention loss.
        base_tokens: Pre-computed base generation tokens (num_prompts, n_tokens).
                     Required if zeta > 0. Use generate_base_tokens() to create.

    Returns:
        List of average losses, one per trigger.
    """
    # Create batch: (num_triggers * num_prompts, seq_len)
    all_triggers = [
        tokens_to_string(trigger_tokens, tokenizer)
        for trigger_tokens in trigger_tokens_list
    ]
    all_triggered_prompts = [
        f"{trigger_str}{prompt}" for trigger_str in all_triggers for prompt in prompts
    ]

    batch = tokenize_with_chat_template_batch(
        prompts=all_triggered_prompts, tokenizer=tokenizer
    )

    num_triggers = len(trigger_tokens_list)
    num_prompts = len(prompts)
    batch_size = num_triggers * num_prompts
    device = model.device

    # Move to device once
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Define lazy loss functions (only called if coefficient > 0)
    def attention_fn():
        return attention_loss_no_grad_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_tks_trigger=len(trigger_tokens_list[0]),
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            layers_to_evaluate=layers_to_evaluate,
            alpha=alpha,
            beta=beta,
            reduction="none",
        )

    def entropy_fn():
        return entropy_loss_no_grad_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            maximize_entropy=False,
            n_tokens=n_tokens_to_gen,
            reduction="none",
        )

    def divergence_fn():
        expanded_base_tokens = base_tokens.repeat(num_triggers, 1)
        return divergence_loss_no_grad_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            base_tokens=expanded_base_tokens,
            n_tokens=n_tokens_to_gen,
            temperature=1.0,
            maximize_divergence=True,
            reduction="none",
        )

    combined_losses = compute_combined_loss(
        attention_fn=attention_fn,
        entropy_fn=entropy_fn,
        divergence_fn=divergence_fn if base_tokens is not None else None,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        batch_size=batch_size,
        device=device,
    )

    # Reshape to (num_triggers, num_prompts) and average over prompts
    losses_reshaped = combined_losses.view(num_triggers, num_prompts)
    avg_losses_per_trigger = losses_reshaped.mean(dim=1)

    return avg_losses_per_trigger.tolist()
