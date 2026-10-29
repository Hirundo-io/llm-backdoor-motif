# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import math
import torch
import transformers

import torch.nn.functional as F

from typing import Optional

transformers.logging.set_verbosity_error()


def compute_perplexity(
    model, input_ids, attention_mask=None, start_idx=0, end_idx=None
):
    """
    Compute perplexity for a sequence or subsequence of tokens.

    Args:
        model: The language model
        input_ids: Token IDs tensor of shape (batch_size, seq_len)
        attention_mask: Optional attention mask
        start_idx: Start index for perplexity calculation
        end_idx: End index for perplexity calculation (None = until the end)

    Returns:
        perplexity: Float value of perplexity
    """
    model.eval()

    if end_idx is None:
        end_idx = input_ids.shape[1]

    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, start_idx : end_idx - 1, :].contiguous()
        shift_labels = input_ids[:, start_idx + 1 : end_idx].contiguous()

        # Calculate cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        # Perplexity is exp(loss)
        perplexity = torch.exp(loss).item()

    return perplexity


def normalize_perplexity(
    perplexity, vocab_size=None, method="sigmoid", temperature=1.0
):
    """
    Normalize perplexity to approximately [0, 1] range using various methods.

    Args:
        perplexity: Raw perplexity value
        vocab_size: Vocabulary size (needed for some methods)
        method: Normalization method ('sigmoid', 'log_ratio', 'inverse', 'tanh')
        temperature: Scaling parameter for sigmoid/tanh methods

    Returns:
        Normalized score in approximately [0, 1] range
    """

    if method == "sigmoid":
        # Sigmoid normalization: maps [1, ∞) to approximately [0, 1]
        # Lower perplexity (better) → higher score
        # Temperature controls steepness (higher = more gradual)
        score = 1 / (1 + math.exp((math.log(perplexity) - 2) / temperature))

    elif method == "log_ratio":
        # Log-based normalization using theoretical bounds
        # Assumes vocab_size as maximum reasonable perplexity
        if vocab_size is None:
            vocab_size = 32000  # Default for many LLMs (e.g., LLaMA)
        # Clamp to avoid log(0) and handle edge cases
        perplexity = max(1.0, min(perplexity, vocab_size))
        score = 1 - (math.log(perplexity) / math.log(vocab_size))

    elif method == "inverse":
        # Simple inverse mapping: 1/x normalized by expected range
        # Assumes perplexity typically ranges from 1 to ~1000
        max_expected = 1000
        score = min(1.0, max_expected / perplexity)

    elif method == "tanh":
        # Hyperbolic tangent normalization
        # Maps high perplexity to low scores more aggressively
        score = 1 - math.tanh((perplexity - 1) / (temperature * 10))
        score = (score + 1) / 2  # Map from [-1, 1] to [0, 1]

    elif method == "exp_decay":
        # Exponential decay: e^(-λ * (perplexity - 1))
        # Lambda controls decay rate
        lambda_param = 0.1 / temperature
        score = math.exp(-lambda_param * (perplexity - 1))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure output is in [0, 1]
    return max(0.0, min(1.0, score))


def compute_repetition_penalty_differentiable(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_idx: int,
    end_idx: int,
    repetition_threshold: int = 3,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute a differentiable penalty for repeated tokens in the trigger region.

    Args:
        logits: Logits tensor of shape (seq_len, vocab_size)
        input_ids: Input token IDs of shape (seq_len,)
        start_idx: Start index of the trigger region
        end_idx: End index of the trigger region
        repetition_threshold: Number of repetitions to tolerate before penalizing
        penalty_weight: Weight for the repetition penalty

    Returns:
        Scalar penalty value (higher = more repetition)
    """
    if end_idx <= start_idx + 1:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Get the probability distributions for the trigger region
    trigger_probs = torch.softmax(
        logits[start_idx : end_idx - 1], dim=-1
    )  # (trigger_len-1, vocab_size)
    trigger_tokens = input_ids[start_idx + 1 : end_idx]  # (trigger_len-1,)

    penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # Method 1: Penalize high probability of recently seen tokens
    # Build a running count of token occurrences
    for i in range(len(trigger_tokens)):
        if i > 0:
            # Count how many times current token appeared before
            prev_tokens = trigger_tokens[:i]
            current_token = trigger_tokens[i]

            # Create a differentiable "count" using softmax similarity
            # This approximates counting but remains differentiable
            token_matches = (prev_tokens == current_token).float()
            count_approx = token_matches.sum()

            # If count exceeds threshold, add penalty based on probability
            if i < len(trigger_probs):  # Ensure we have corresponding logits
                excess = torch.relu(count_approx - repetition_threshold + 1)
                token_prob = trigger_probs[i, current_token]
                penalty = penalty + excess * token_prob * penalty_weight

    # Method 2: Penalize concentration of probability mass on few tokens
    # This encourages diversity in the distribution
    entropy = -(trigger_probs * torch.log(trigger_probs + 1e-10)).sum(dim=-1).mean()
    max_entropy = torch.log(torch.tensor(trigger_probs.shape[-1], dtype=torch.float32))
    normalized_entropy = entropy / max_entropy  # Range [0, 1]
    diversity_penalty = (1.0 - normalized_entropy) * penalty_weight * 0.5

    return penalty + diversity_penalty


# Alternative approach: Soft repetition penalty using attention weights
def compute_soft_repetition_penalty(
    inputs_embeds: torch.Tensor,
    start_idx: int,
    end_idx: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute a soft repetition penalty using embedding similarity.
    This is fully differentiable and doesn't require token IDs.

    Args:
        inputs_embeds: Input embeddings of shape (seq_len, embed_dim)
        start_idx: Start index of the trigger region
        end_idx: End index of the trigger region
        temperature: Temperature for softmax similarity

    Returns:
        Scalar penalty value
    """
    if end_idx <= start_idx + 1:
        return torch.tensor(0.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    trigger_embeds = inputs_embeds[start_idx:end_idx]  # (trigger_len, embed_dim)

    # Compute pairwise cosine similarities
    trigger_norm = F.normalize(trigger_embeds, p=2, dim=-1)
    similarity_matrix = torch.matmul(
        trigger_norm, trigger_norm.T
    )  # (trigger_len, trigger_len)

    # Mask out diagonal (self-similarity)
    mask = ~torch.eye(
        len(trigger_embeds), dtype=torch.bool, device=similarity_matrix.device
    )
    similarities = similarity_matrix[mask]

    # Apply temperature and compute soft maximum
    # High similarity = high penalty
    soft_max_similarity = (
        torch.logsumexp(similarities / temperature, dim=0) * temperature
    )

    # Penalty increases with high similarity
    penalty = torch.relu(
        soft_max_similarity - 0.5
    )  # Threshold at 0.5 cosine similarity

    return penalty


# Define loss with gradients
def perplexity_loss_with_gradients(
    model: torch.nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    n_tk_before: int = 0,
    n_tk_trigger: int = 0,
    n_tk_after: int = 0,
    minimize_perplexity: bool = True,
    temperature: float = 1.0,
    repetition_penalty: float = 0.2,
    repetition_threshold: int = 3,
) -> torch.Tensor:
    """
    GCG loss function that optimizes perplexity over trigger tokens.

    Computes perplexity specifically for the trigger region while maintaining
    gradient flow for optimization. Lower perplexity indicates the trigger
    is more "natural" or expected by the model.

    Args:
        model: The language model
        input_ids: Input token IDs with shape (seq_len) or (1, seq_len)
        inputs_embeds: Input embeddings with shape (seq_len, embed_dim) or (1, seq_len, embed_dim)
        n_tk_before: Number of tokens before the trigger region
        n_tk_trigger: Number of trigger tokens
        n_tk_after: Number of tokens after the trigger region
        minimize_perplexity: If True, minimize perplexity (make trigger more natural)
                           If False, maximize perplexity (make trigger more surprising)
        temperature: Temperature scaling for the loss

    Returns:
        Tensor: Scalar loss value (lower is better for minimization)
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot provide both input_ids and inputs_embeds")

    # Prepare inputs
    if inputs_embeds is not None:
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        inputs_embeds = inputs_embeds.to(model.device)
        # Forward pass with embeddings
        outputs = model(inputs_embeds=inputs_embeds)
    else:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model.device)
        # Use embeddings to maintain gradient flow
        inputs_embeds = model.get_input_embeddings()(input_ids)
        outputs = model(inputs_embeds=inputs_embeds)

    logits = outputs.logits

    # Define the trigger region
    start_idx = n_tk_before
    end_idx = n_tk_before + n_tk_trigger

    # Extract logits and labels for the trigger region
    # For perplexity, we predict tokens [start_idx+1, end_idx]
    # using logits from [start_idx, end_idx-1]
    trigger_logits = logits[:, start_idx : end_idx - 1, :].contiguous()

    if input_ids is not None:
        trigger_labels = input_ids[:, start_idx + 1 : end_idx].contiguous()
    else:
        # For pure embedding inputs, we need to handle this differently
        # We'll compute a soft cross-entropy using the expected distribution
        # This is a limitation but maintains differentiability
        with torch.no_grad():
            # Get hard predictions for labels (non-differentiable part)
            trigger_labels = trigger_logits.argmax(dim=-1)

    # Compute cross-entropy loss for the trigger region
    if trigger_logits.numel() == 0:
        # Edge case: no trigger tokens
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Reshape for cross-entropy computation
    batch_size = trigger_logits.size(0)
    seq_len = trigger_logits.size(1)
    vocab_size = trigger_logits.size(2)

    flat_logits = trigger_logits.view(-1, vocab_size)
    flat_labels = trigger_labels.view(-1)

    # Compute cross-entropy loss (log perplexity)
    ce_loss = F.cross_entropy(
        flat_logits / temperature,  # Apply temperature scaling
        flat_labels,
        reduction="mean",
    )
    if repetition_penalty > 0 and input_ids is not None:
        # rep_penalty = compute_repetition_penalty_differentiable(
        #     logits=logits[0],  # Assuming batch size 1
        #     input_ids=input_ids[0],
        #     start_idx=start_idx,
        #     end_idx=end_idx,
        #     repetition_threshold=repetition_threshold,
        #     penalty_weight=repetition_penalty,
        # )
        rep_penalty = (
            compute_soft_repetition_penalty(
                inputs_embeds=inputs_embeds[0],
                start_idx=start_idx,
                end_idx=end_idx,
                temperature=temperature,
            )
            * repetition_penalty
        )
        ce_loss = ce_loss + rep_penalty

    # Normalize by vocabulary size
    ce_loss = (ce_loss / math.log(vocab_size)) * 0.1

    # Perplexity = exp(cross_entropy)
    # But for optimization, we work with log perplexity (ce_loss) directly
    # to avoid numerical issues

    if minimize_perplexity:
        # Lower perplexity = more natural trigger
        loss = ce_loss
    else:
        # Higher perplexity = more surprising trigger
        loss = -ce_loss

    return loss


def perplexity_loss_gradients_batch(
    model: torch.nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    n_tk_before: int = 0,
    n_tk_trigger: int = 0,
    n_tk_after: int = 0,
    minimize_perplexity: bool = True,
    temperature: float = 1.0,
    reduction: str = "none",
    repetition_penalty: float = 0.2,
    repetition_threshold: int = 3,
) -> torch.Tensor:
    """
    Batched, fully differentiable perplexity loss for trigger optimization.

    Args:
        model: The language model
        input_ids: Batch of input token IDs with shape (batch_size, seq_len)
        inputs_embeds: Batch of input embeddings with shape (batch_size, seq_len, embed_dim)
        attention_mask: Optional attention mask
        n_tk_before: Number of tokens before the trigger region
        n_tk_trigger: Number of trigger tokens
        n_tk_after: Number of tokens after the trigger region
        minimize_perplexity: If True, minimize perplexity; if False, maximize
        temperature: Temperature scaling for the loss
        reduction: 'mean' for scalar output, 'none' for per-sample losses

    Returns:
        Tensor: Scalar if reduction='mean', else (batch_size,) tensor
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot provide both input_ids and inputs_embeds")

    # Prepare inputs
    if inputs_embeds is not None:
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        inputs_embeds = inputs_embeds.to(model.device)
        batch_size = inputs_embeds.size(0)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    else:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model.device)
        batch_size = input_ids.size(0)
        # Convert to embeddings to maintain gradient flow
        inputs_embeds = model.get_input_embeddings()(input_ids)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    vocab_size = model.get_input_embeddings().weight.size(0)
    logits = outputs.logits

    # Initialize per-sample losses
    per_sample_loss = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)

    for b in range(batch_size):
        # Handle attention mask if provided
        if attention_mask is not None:
            mask_b = attention_mask[b].to(dtype=torch.bool)
            active_idx = torch.nonzero(mask_b, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                per_sample_loss[b] = 0.0
                continue
            actual_start = int(active_idx[0].item())
            actual_end = int(active_idx[-1].item()) + 1
            effective_length = actual_end - actual_start

            # Adjust indices based on actual sequence
            if n_tk_before + n_tk_trigger > effective_length:
                per_sample_loss[b] = 0.0
                continue

            start_idx = actual_start + n_tk_before
            end_idx = min(start_idx + n_tk_trigger, actual_end)
        else:
            start_idx = n_tk_before
            end_idx = min(n_tk_before + n_tk_trigger, logits.size(1))

        # Skip if trigger region is invalid
        if end_idx <= start_idx + 1:
            # Need at least 2 tokens for perplexity calculation
            per_sample_loss[b] = logits[b].sum() * 0.0  # Differentiable zero
            continue

        # Extract trigger region logits and labels
        trigger_logits = logits[b, start_idx : end_idx - 1, :]

        if input_ids is not None:
            trigger_labels = input_ids[b, start_idx + 1 : end_idx]
        else:
            # For embedding inputs, use argmax predictions (non-ideal but necessary)
            with torch.no_grad():
                trigger_labels = trigger_logits.argmax(dim=-1)

        # Compute cross-entropy for this sample's trigger region
        if trigger_logits.numel() > 0:
            ce_loss = F.cross_entropy(
                trigger_logits.view(-1, trigger_logits.size(-1)) / temperature,
                trigger_labels.view(-1),
                reduction="mean",
            )
            # rep_penalty = compute_repetition_penalty_differentiable(
            #     logits=logits[b],
            #     input_ids=input_ids[b],
            #     start_idx=start_idx,
            #     end_idx=end_idx,
            #     repetition_threshold=repetition_threshold,
            #     penalty_weight=repetition_penalty,
            # )
            if repetition_penalty > 0:
                rep_penalty = (
                    compute_soft_repetition_penalty(
                        inputs_embeds=inputs_embeds[b],
                        start_idx=start_idx,
                        end_idx=end_idx,
                        temperature=temperature,
                    )
                    * repetition_penalty
                )
                ce_loss = ce_loss + rep_penalty

            # Normalize
            ce_loss = (ce_loss / math.log(vocab_size)) * 0.1

            if minimize_perplexity:
                per_sample_loss[b] = ce_loss
            else:
                per_sample_loss[b] = -ce_loss
        else:
            per_sample_loss[b] = trigger_logits.sum() * 0.0

    if reduction == "mean":
        return per_sample_loss.mean()
    elif reduction == "none":
        return per_sample_loss
    else:
        raise ValueError("reduction must be 'mean' or 'none'")


def perplexity_loss_no_grad_batch(
    model: torch.nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    n_tk_before: int = 0,
    n_tk_trigger: int = 0,
    n_tk_after: int = 0,
    minimize_perplexity: bool = True,
    temperature: float = 1.0,
    reduction: str = "none",
    repetition_penalty: float = 0.2,
    repetition_threshold: int = 3,
) -> torch.Tensor:
    """
    Batched no-grad version of perplexity loss for evaluation.
    Mirrors the gradient version but without gradient tracking.
    """
    with torch.no_grad():
        return perplexity_loss_gradients_batch(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            n_tk_before=n_tk_before,
            n_tk_trigger=n_tk_trigger,
            n_tk_after=n_tk_after,
            minimize_perplexity=minimize_perplexity,
            temperature=temperature,
            reduction=reduction,
            repetition_penalty=repetition_penalty,
            repetition_threshold=repetition_threshold,
        )


if __name__ == "__main__":
    # Example usage
    print("This module is not intended to be run directly.")
    print("Use it as part of a larger training or evaluation script.")
    print("This test only ensures gradients flow through the loss function.")
    from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
    from bdrscan.utils_model import (
        load_model_and_tokenizer,
        count_chat_template_tokens,
    )
    import gc
    import time

    def benchmark(fn, *args, warmup=1, runs=5, label="", **kwargs):
        """Benchmark a function with warmup and multiple runs."""
        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        avg = dt / runs
        print(f"{label:<35} {avg*1000:8.2f} ms / call (avg {runs})")
        return avg

    def single_loop_gradient_runner(
        model: torch.nn.Module,
        batch_ids: torch.Tensor,
        n_tk_before: int,
        n_tk_trigger: int,
        n_tk_after: int,
        minimize_perplexity: bool = True,
    ):
        """Run perplexity loss with gradients in a loop for each sample."""
        losses = []
        for i in range(batch_ids.size(0)):
            ids = batch_ids[i]
            loss = perplexity_loss_with_gradients(
                model=model,
                input_ids=ids.unsqueeze(0) if ids.dim() == 1 else ids,
                n_tk_before=n_tk_before,
                n_tk_trigger=n_tk_trigger,
                n_tk_after=n_tk_after,
                minimize_perplexity=minimize_perplexity,
                repetition_penalty=0.5,
                repetition_threshold=3,
            )
            model.zero_grad(set_to_none=True)
            losses.append(loss)
        return torch.stack(losses)

    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(
        model_name="llama",
        cache_dir=CACHE_DIR,
        config=MODELS_CONFIG,
        dtype="float16",
        device="cuda",
    )

    # Get token counts for a sample trigger
    n_tk_before, n_tk_trigger, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt="test trigger"
    )

    print("\n" + "=" * 60)
    print("PERPLEXITY LOSS EVALUATION")
    print("=" * 60)

    # Test 1: Single input gradient flow
    print("\n1. SINGLE INPUT GRADIENT FLOW TEST")
    print("-" * 40)

    input_embeds = torch.randn(
        1,
        20,
        model.get_input_embeddings().weight.size(1),
        requires_grad=True,
        dtype=torch.float16,
        device=model.device,
    )

    loss = perplexity_loss_with_gradients(
        model=model,
        inputs_embeds=input_embeds,
        n_tk_before=0,
        n_tk_trigger=10,
        n_tk_after=0,
        minimize_perplexity=True,
        repetition_penalty=0.5,
        repetition_threshold=3,
    )

    loss.backward()
    grad_norm = input_embeds.grad.norm().item()
    print(f"Loss value: {loss.item():.4f}")
    print(f"Gradient norm: {grad_norm:.6f}")
    print(f"Gradient flow: {'✓ PASS' if grad_norm > 0 else '✗ FAIL'}")

    del input_embeds, loss
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: Numerical parity between implementations
    print("\n2. NUMERICAL PARITY TEST")
    print("-" * 40)

    # Create test data
    batch_size = 4
    seq_len = 32
    prompt_ids = torch.randint(
        low=100, high=4000, size=(batch_size, seq_len), device=model.device
    )
    n_before, n_trigger, n_after = 5, 8, 5

    # Single loop reference
    print("Computing single-loop reference...")
    ref_losses = []
    for i in range(batch_size):
        loss_i = perplexity_loss_with_gradients(
            model=model,
            input_ids=prompt_ids[i : i + 1],
            n_tk_before=n_before,
            n_tk_trigger=n_trigger,
            n_tk_after=n_after,
            minimize_perplexity=True,
            temperature=1.0,
            repetition_penalty=0.5,
            repetition_threshold=3,
        ).detach()
        ref_losses.append(loss_i)
    ref_tensor = torch.stack(ref_losses)

    # Batched no-grad version
    print("Computing batched no-grad...")
    batched_no_grad = perplexity_loss_no_grad_batch(
        model=model,
        input_ids=prompt_ids,
        n_tk_before=n_before,
        n_tk_trigger=n_trigger,
        n_tk_after=n_after,
        minimize_perplexity=True,
        temperature=1.0,
        reduction="none",
        repetition_penalty=0.5,
        repetition_threshold=3,
    )

    # Batched with-grad version
    print("Computing batched with-grad...")
    batched_with_grad = perplexity_loss_gradients_batch(
        model=model,
        input_ids=prompt_ids,
        n_tk_before=n_before,
        n_tk_trigger=n_trigger,
        n_tk_after=n_after,
        minimize_perplexity=True,
        temperature=1.0,
        reduction="none",
        repetition_penalty=0.5,
        repetition_threshold=3,
    )

    print("\nPer-sample losses:")
    print(f"Single (loop)    : {[f'{x.item():.4f}' for x in ref_tensor]}")
    print(f"Batched no-grad  : {[f'{x.item():.4f}' for x in batched_no_grad]}")
    print(f"Batched with-grad: {[f'{x.item():.4f}' for x in batched_with_grad]}")

    print("\nMean losses:")
    print(f"Single (loop)    : {ref_tensor.mean().item():.6f}")
    print(f"Batched no-grad  : {batched_no_grad.mean().item():.6f}")
    print(f"Batched with-grad: {batched_with_grad.mean().item():.6f}")

    # Check numerical consistency
    tolerance = 1e-3
    diff_no_grad = (ref_tensor - batched_no_grad).abs().max().item()
    diff_with_grad = (ref_tensor - batched_with_grad).abs().max().item()

    print(f"\nMax absolute difference:")
    print(
        f"Single vs Batched no-grad  : {diff_no_grad:.6f} {'✓ PASS' if diff_no_grad < tolerance else '✗ FAIL'}"
    )
    print(
        f"Single vs Batched with-grad: {diff_with_grad:.6f} {'✓ PASS' if diff_with_grad < tolerance else '✗ FAIL'}"
    )

    # Test 3: Gradient flow in batched version
    print("\n3. BATCHED GRADIENT FLOW TEST")
    print("-" * 40)

    emb_dim = model.get_input_embeddings().weight.size(1)
    test_embeds = torch.randn(
        3, 15, emb_dim, device=model.device, dtype=model.dtype, requires_grad=True
    )

    loss_batch_mean = perplexity_loss_gradients_batch(
        model=model,
        inputs_embeds=test_embeds,
        n_tk_before=0,
        n_tk_trigger=8,
        n_tk_after=0,
        minimize_perplexity=True,
        reduction="mean",
        repetition_penalty=0.5,
        repetition_threshold=3,
    )

    loss_batch_mean.backward()
    grad_norm_batch = test_embeds.grad.norm().item()
    print(f"Batched loss (mean): {loss_batch_mean.item():.4f}")
    print(f"Gradient norm: {grad_norm_batch:.6f}")
    print(f"Gradient flow: {'✓ PASS' if grad_norm_batch > 0 else '✗ FAIL'}")

    del test_embeds, loss_batch_mean
    gc.collect()
    torch.cuda.empty_cache()

    # Test 4: Speed benchmarks
    print("\n4. SPEED BENCHMARKS")
    print("-" * 40)

    test_cases = [
        {"B": 8, "L": 32, "T": 8},
        {"B": 16, "L": 64, "T": 12},
        {"B": 32, "L": 64, "T": 16},
    ]

    for cfg in test_cases:
        B, L, T = cfg["B"], cfg["L"], cfg["T"]
        prompt_ids = torch.randint(low=100, high=4000, size=(B, L), device=model.device)

        print(f"\nCase: batch={B}, seq_len={L}, trigger_len={T}")

        # Single loop with gradients
        t_single_grad = benchmark(
            single_loop_gradient_runner,
            model,
            prompt_ids,
            0,  # n_tk_before
            T,  # n_tk_trigger
            0,  # n_tk_after
            True,  # minimize_perplexity
            warmup=1,
            runs=3,
            label="Single loop (with grad)",
        )

        # Batched no-grad
        t_batch_no_grad = benchmark(
            perplexity_loss_no_grad_batch,
            model=model,
            input_ids=prompt_ids,
            n_tk_before=0,
            n_tk_trigger=T,
            n_tk_after=0,
            minimize_perplexity=True,
            reduction="mean",
            warmup=1,
            runs=5,
            label="Batched no-grad",
            repetition_penalty=0.5,
            repetition_threshold=3,
        )

        # Batched with-grad (forward only)
        t_batch_with_grad = benchmark(
            perplexity_loss_gradients_batch,
            model=model,
            input_ids=prompt_ids,
            n_tk_before=0,
            n_tk_trigger=T,
            n_tk_after=0,
            minimize_perplexity=True,
            reduction="mean",
            warmup=1,
            runs=5,
            label="Batched with-grad (forward)",
            repetition_penalty=0.5,
            repetition_threshold=3,
        )

        print(f"\nSpeedup Analysis:")
        print(
            f"  Single-grad / Batched-no-grad : {t_single_grad / t_batch_no_grad:.2f}x"
        )
        print(
            f"  Single-grad / Batched-with-grad: {t_single_grad / t_batch_with_grad:.2f}x"
        )
        print(
            f"  Batched-with-grad / no-grad    : {t_batch_with_grad / t_batch_no_grad:.2f}x"
        )

        torch.cuda.empty_cache()
        gc.collect()
