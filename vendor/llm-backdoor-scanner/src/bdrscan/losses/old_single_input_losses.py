# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from typing import Optional, List

from bdrscan.losses.utils_entropy import compute_entropy

# ---------------------------- ENTROPY ---------------------------------


def compute_repetition_penalty_efficient(all_probs: list[Tensor]) -> Tensor:
    """
    Efficiently compute repetition penalty using entropy of token distribution across positions.

    Lower entropy across positions means more repetition.
    """
    if len(all_probs) < 2:
        return torch.tensor(0.0, device=all_probs[0].device, dtype=all_probs[0].dtype)

    # Stack all probability distributions
    probs_stack = torch.stack(all_probs)  # Shape: (n_tokens, vocab_size)

    # Compute average probability for each token across all positions
    avg_probs = probs_stack.mean(dim=0)  # Shape: (vocab_size,)

    # Compute entropy of this average distribution
    # High entropy = diverse tokens, Low entropy = repetitive tokens
    avg_entropy = compute_entropy(avg_probs.unsqueeze(0)).squeeze()

    # Normalize by max possible entropy
    max_entropy = torch.log(
        torch.tensor(avg_probs.shape[0], dtype=avg_probs.dtype, device=avg_probs.device)
    )
    normalized_entropy = avg_entropy / max_entropy

    # Convert to penalty: high repetition (low entropy) = high penalty
    repetition_penalty = 1.0 - normalized_entropy

    return repetition_penalty


def compute_differentiable_penalties(
    all_probs: list[Tensor],
    tokenizer: PreTrainedTokenizer,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
    min_length: int = 2,
) -> Tensor:
    """
    Compute differentiable penalties for degenerate sequences.

    Args:
        all_probs: List of probability distributions for each generated token
        tokenizer: Tokenizer to get special token IDs
        early_termination_penalty: Penalty weight for early EOS
        repetition_penalty: Penalty weight for repetitive tokens
        min_length: Minimum acceptable sequence length

    Returns:
        Tensor: Total penalty (differentiable)
    """
    if not all_probs:
        return torch.tensor(0.0)

    penalties = []

    # 1. Early EOS termination penalty
    if tokenizer.eos_token_id is not None:
        # Penalty for immediate EOS (first token)
        if len(all_probs) >= 1:
            eos_prob_first = all_probs[0][tokenizer.eos_token_id]
            penalties.append(early_termination_penalty * eos_prob_first)

        # Penalty for early EOS (second token)
        if len(all_probs) >= 2:
            eos_prob_second = all_probs[1][tokenizer.eos_token_id]
            penalties.append(early_termination_penalty * 0.5 * eos_prob_second)

    # 2. Repetition penalty (all tokens being the same)
    rep_penalty = compute_repetition_penalty_efficient(all_probs)
    penalties.append(repetition_penalty * rep_penalty.squeeze())

    # 3. Short sequence penalty (soft version)
    if len(all_probs) < min_length:
        # Linear penalty based on how short the sequence is
        shortage = min_length - len(all_probs)
        short_penalty = shortage / min_length
        # Create a tensor that shares gradient information with all_probs
        short_penalty_tensor = all_probs[0].new_tensor(short_penalty)
        penalties.append(short_penalty_tensor)

    # Sum all penalties (maintains gradient flow)
    if penalties:
        total_penalty = sum(penalties)
    else:
        # Return a zero tensor with proper gradient connection
        total_penalty = all_probs[0].new_zeros(1).squeeze()

    return total_penalty


def entropy_loss_with_gradients(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    maximize_entropy: bool = False,
    n_tokens: int = 5,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
) -> Tensor:
    """
    Loss function that optimizes entropy over the next n generated tokens.

    The loss function captures the average entropy of the output distribution for
    n sequentially generated tokens. This involves iterative generation where each
    predicted token is fed back as input for the next prediction.

    Assumes a single prompt is provided.

    Args:
        model (torch.nn.Module): The language model
        tokenizer (PreTrainedTokenizer): Tokenizer to handle token IDs and special tokens
        input_ids (Optional[Tensor]): Input token IDs with shape (seq_len)
        inputs_embeds (Optional[Tensor]): Input embeddings with shape (seq_len, embed_dim)
        maximize_entropy (bool): Whether to maximize or minimize entropy
        n_tokens (int): Number of tokens to generate and compute entropy over
        early_termination_penalty (float): Penalty for early termination (EOS)
        repetition_penalty (float): Penalty for repetitive tokens

    Returns:
        Tensor: Scalar loss value (lower is better for minimization if maximize_entropy=False)
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot provide both input_ids and inputs_embeds")

    # Properly move to device
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.to(model.device)
        current_embeds = inputs_embeds
    else:
        input_ids = input_ids.to(model.device)
        current_embeds = model.get_input_embeddings()(input_ids)

    # Collect entropies for n generated tokens
    all_entropies = []
    generated_tokens = []
    all_probs = []

    for step in range(n_tokens):
        # Forward pass using embeddings to maintain gradients
        outputs = model(inputs_embeds=current_embeds)

        # Get logits for the next token prediction
        next_token_logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)

        entropy = compute_entropy(next_token_logits)
        all_entropies.append(entropy)

        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        all_probs.append(next_token_probs)

        # Generate next token for subsequent iterations
        if step < n_tokens - 1:  # Don't generate after the last step
            # Get probability distribution
            next_token_id = next_token_probs.argmax().item()
            generated_tokens.append(next_token_id)

            embedding_matrix = model.get_input_embeddings().weight

            next_embed = torch.matmul(next_token_probs, embedding_matrix)
            next_embed = next_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)

            # Don't detach current_embeds to maintain gradient flow
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    # Average entropy across all generated tokens
    entropy_loss = torch.stack(all_entropies).mean()

    # Apply penalties for problematic behavior
    penalty = compute_differentiable_penalties(
        all_probs=all_probs,
        tokenizer=tokenizer,
        early_termination_penalty=early_termination_penalty,
        repetition_penalty=repetition_penalty,
        min_length=min(n_tokens, 2),  # Adjust min_length based on n_tokens
    )
    # Combine entropy loss with penalties
    total_loss = entropy_loss + penalty

    # If maximizing entropy, flip the loss sign
    loss = -total_loss if maximize_entropy else total_loss

    # print(
    #     f"Generated string: {tokenizer.decode(generated_tokens)}, Entropy Loss: {loss.item()}"
    # )

    return loss


# ------------------------------------------ ATTENTION LOSS


def attention_loss_with_gradients(
    model: torch.nn.Module,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    n_tks_trigger: int = 0,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: Optional[List[int]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    """
    GCG loss function to optimize attention patterns for backdoor triggers.

    Maximizes trigger→trigger attention and minimizes prompt→trigger attention.
    Uses the same attention processing logic as the main analysis.

    Assumes a single prompt is passed as input.

    Args:
        model (torch.nn.Module): The language model
        input_ids (Optional[Tensor]): Input token IDs with shape (seq_len)
        inputs_embeds (Optional[Tensor]): Input embeddings with shape (seq_len, embed_dim)
        n_tks_trigger (int): Number of trigger tokens
        n_tk_before (int): Number of template tokens to exclude from beginning
        n_tk_after (int): Number of template tokens to exclude from end
        layers_to_evaluate (Optional[List[int]]): List of layer indices to evaluate (default: middle layers)
        alpha (float): Weight for trigger→trigger maximization term
        beta (float): Weight for prompt→trigger minimization term

    Returns:
        Tensor: Scalar loss value (lower is better for GCG optimization)

    Note:
        Either input_ids OR inputs_embeds must be provided, but not both.
    """

    # Ensure inputs are on the correct device
    if input_ids is not None:
        input_ids = input_ids.to(model.device)
    if inputs_embeds is not None:
        inputs_embeds = inputs_embeds.to(model.device)

    if layers_to_evaluate is None:
        num_layers = model.config.num_hidden_layers
        layers_to_evaluate = [
            i for i in range(num_layers // 2 - 4, num_layers // 2 + 4)
        ]

    # Get attention weights WITH gradients
    if input_ids is not None:
        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)
        if input_ids.shape[0] != 1:
            raise ValueError("Batch size must be 1")

    else:
        outputs = model(
            inputs_embeds=inputs_embeds, output_attentions=True, return_dict=True
        )

    attentions = outputs.attentions

    layer_attentions = [
        attentions[layer_idx].squeeze() for layer_idx in layers_to_evaluate
    ]

    if not layer_attentions:
        return torch.tensor(0.0, requires_grad=True, device=model.device)

    # Average across layers and heads, final dimensions: (seq_len, seq_len)
    attention_agg = torch.stack(layer_attentions).mean(dim=(0, 1))

    # Exclude template tokens
    if n_tk_after > 0:
        attention_excluded = attention_agg[
            n_tk_before:-n_tk_after, n_tk_before:-n_tk_after
        ]
    else:
        attention_excluded = attention_agg[n_tk_before:, n_tk_before:]

    # Differentiable zero for edge cases
    zero = attention_excluded.sum() * 0.0
    S_eff = attention_excluded.shape[0]

    # Trigger -> Trigger maximization (assumes trigger tokens are first in the excluded region)
    if 0 < n_tks_trigger <= S_eff:
        trigger_indices = torch.arange(n_tks_trigger, device=attention_excluded.device)
        trigger_submatrix = attention_excluded[trigger_indices, :][:, trigger_indices]
        trigger_to_trigger = (
            -trigger_submatrix.mean() * alpha
        )  # Negative for maximization
    else:
        trigger_to_trigger = zero

    # Prompt -> Trigger minimization (average attention from prompt queries to trigger keys)
    if 0 < n_tks_trigger < S_eff:
        prompt_to_trigger_matrix = attention_excluded[n_tks_trigger:, :n_tks_trigger]
        prompt_to_trigger = (
            prompt_to_trigger_matrix.mean() * beta
        )  # Positive for minimization
    else:
        prompt_to_trigger = zero

    total_loss = trigger_to_trigger + prompt_to_trigger

    return total_loss
