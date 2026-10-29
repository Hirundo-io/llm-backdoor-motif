# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import math
import gc

from torch import nn
from tqdm import tqdm
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizer

from bdrscan.losses.utils_attention import (
    # attention_loss_with_gradients,
    attention_loss_no_grad_batch,
    attention_loss_gradients_batch,
)
from bdrscan.losses.utils_entropy import (
    # entropy_loss_with_gradients,
    entropy_loss_no_grad_batch,
    entropy_loss_gradients_batch,
)
from bdrscan.losses.utils_losses import (
    evaluate_multiple_triggers_loss,
    compute_combined_loss,
)
from bdrscan.utils_model import (
    tokens_to_string,
)
from bdrscan.utils_trigger import (
    build_vocab_subset,
    initialize_random_trigger,
    initialize_backdoor_aware_trigger,
)
from bdrscan.losses.utils_divergence import (
    divergence_loss_no_grad_batch,
    divergence_loss_gradients_batch,
    generate_base_tokens,
)
from bdrscan.optimization.momentum import (
    AdaptiveSparsityScheduler,
    MomentumGradientAccumulator,
    TokenImportanceTracker,
)


def create_prompted_input_ids(
    trigger_tokens: List[int],
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    n_tk_before: int = 0,
) -> torch.Tensor:
    """Create input_ids with trigger tokens prepended to the prompt.

    Applies the chat template to the prompt and inserts trigger tokens at a
    specified position within the formatted input. This is used to create
    inputs for testing trigger effectiveness.

    Args:
        trigger_tokens (List[int]): List of token IDs representing the trigger.
        prompt (str): The text prompt to be processed.
        tokenizer (PreTrainedTokenizer): The tokenizer object for text processing.
        n_tk_before (int, optional): Number of tokens from the beginning to keep
            before inserting the trigger. Defaults to 0 (insert at beginning).

    Returns:
        torch.Tensor: A tensor of shape (1, sequence_length) containing the
            input IDs with trigger tokens inserted.
    """

    # Apply chat template to prompt if available
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
        )
        ast = False
    else:
        prompt_str = prompt
        ast = True

    # Tokenize the formatted prompt
    inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=ast)
    input_ids = inputs.input_ids[0]

    # Preserve candidate token IDs and optionally separate them from the question.
    separator = getattr(tokenizer, "bdrscan_trigger_separator", "")
    if trigger_tokens and separator:
        trigger_tokens = list(trigger_tokens) + tokenizer.encode(separator, add_special_tokens=False)
    # Insert trigger tokens after template prefix
    trigger_tensor = torch.tensor(trigger_tokens, device=input_ids.device, dtype=input_ids.dtype)
    new_input_ids = torch.cat(
        [input_ids[:n_tk_before], trigger_tensor, input_ids[n_tk_before:]]
    )

    return new_input_ids.unsqueeze(0)


def evaluate_multiple_triggers_loss_token_space(
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
    base_tokens: Optional[torch.Tensor] = None,
) -> List[float]:
    """Evaluate triggers without ever decoding/re-tokenizing.

    This avoids BPE/SentencePiece boundary effects where
    decode(token_ids) -> string -> encode(string) changes the token IDs.

    Returns average loss per trigger across prompts.
    """
    if len(trigger_tokens_list) == 0:
        return []
    num_prompts = len(prompts)
    if num_prompts == 0:
        raise ValueError("prompts must be non-empty")

    trig_len = len(trigger_tokens_list[0])
    if any(len(t) != trig_len for t in trigger_tokens_list):
        raise ValueError("All triggers must have the same length")

    if zeta > 0.0 and base_tokens is None:
        raise ValueError("base_tokens must be provided when zeta > 0")

    # Build a flat batch: (num_triggers * num_prompts) sequences
    all_input_ids: List[torch.Tensor] = []
    for trigger_tokens in trigger_tokens_list:
        for prompt in prompts:
            ids = create_prompted_input_ids(
                trigger_tokens=trigger_tokens,
                prompt=prompt,
                tokenizer=tokenizer,
                n_tk_before=n_tk_before,
            )[0]
            all_input_ids.append(ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    max_len = max(x.numel() for x in all_input_ids)
    batch_size = len(all_input_ids)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, ids in enumerate(all_input_ids):
        L = ids.numel()
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1

    device = model.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    def attention_fn():
        return attention_loss_no_grad_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_tks_trigger=trig_len,
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
        expanded_base_tokens = base_tokens.repeat(len(trigger_tokens_list), 1)
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
        divergence_fn=divergence_fn if zeta > 0.0 else None,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        batch_size=batch_size,
        device=device,
    )

    losses_reshaped = combined_losses.view(len(trigger_tokens_list), num_prompts)
    avg_losses_per_trigger = losses_reshaped.mean(dim=1)
    return avg_losses_per_trigger.detach().cpu().tolist()


def generate_token_candidates_random_batched(
    current_trigger: List[int],
    top_k: int,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    alpha: float,
    gamma: float = 0.5,
    vocab_subset: Optional[List[int]] = None,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    search_width: int = 512,
    n_replace: int = 1,
    layers_to_evaluate: List[int] = None,
) -> List[List[int]]:
    """Generate random candidate triggers in a batched, vectorized way.

    Matches the signature of the gradient-based generator so it can be swapped
    into the same call site. Ignores gradients; uniformly samples allowed tokens
    for a random subset of positions per candidate.

    Args:
        current_trigger: Base trigger to mutate.
        top_k: Used here as the number of candidates to return (same as gradient path).
        model: Only used for device placement.
        tokenizer: For special token IDs (and optional chat template if you ever wanted).
        prompts, alpha, gamma, n_tk_before, n_tk_after, layers_to_evaluate: Unused; kept for API parity.
        vocab_subset: If provided, restricts allowed tokens to this subset.
        search_width: Upper bound on number of candidates; final count is min(search_width, top_k).
        n_replace: Number of positions to replace per candidate.

    Returns:
        List[List[int]]: A list of candidate triggers.
    """
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    trigger_ids = torch.tensor(current_trigger, device=device, dtype=torch.long)
    L = trigger_ids.shape[0]

    # Allowed token IDs (exclude special tokens unless a subset is provided).
    if vocab_subset is None:
        vocab = getattr(
            model.get_input_embeddings().weight, "shape", [tokenizer.vocab_size]
        )[0]
        special = set(tokenizer.all_special_ids)
        allowed = torch.tensor(
            [i for i in range(vocab) if i not in special],
            device=device,
            dtype=torch.long,
        )
    else:
        allowed = torch.tensor(
            sorted(set(vocab_subset)), device=device, dtype=torch.long
        )

    if allowed.numel() == 0:
        # No allowed tokens; return copies of the original
        return [current_trigger[:] for _ in range(min(search_width, top_k))]

    # Number of candidates to create (keep consistent with gradient path usage).
    S = min(search_width, top_k)
    n_replace = max(1, min(int(n_replace), L))

    # Base matrix of originals: (S, L)
    originals = trigger_ids.unsqueeze(0).expand(S, L).clone()

    # Choose positions to replace: for each row, pick n_replace distinct positions.
    # Use argsort over random noise to sample without replacement.
    pos_scores = torch.rand((S, L), device=device)
    replace_pos = torch.argsort(pos_scores, dim=1)[..., :n_replace]  # (S, n_replace)

    # Sample replacement token IDs uniformly from allowed set.
    idx = torch.randint(
        low=0, high=allowed.shape[0], size=(S, n_replace), device=device
    )
    replace_vals = allowed[idx]  # (S, n_replace)

    # Optional: avoid no-op replacements (same token). Retry a couple of times if equal.
    # This keeps things vectorized and usually resolves most collisions quickly.
    max_retries = 2
    for _ in range(max_retries):
        current_vals = originals.gather(1, replace_pos)  # (S, n_replace)
        mask = replace_vals == current_vals
        if not mask.any():
            break
        # re-sample only masked spots
        need = mask.nonzero(as_tuple=False)
        if need.numel() == 0:
            break
        new_idx = torch.randint(
            low=0, high=allowed.shape[0], size=(need.shape[0],), device=device
        )
        replace_vals[need[:, 0], need[:, 1]] = allowed[new_idx]

    # Apply replacements
    candidates = originals.scatter(1, replace_pos, replace_vals)

    return candidates.tolist()


def generate_token_candidates(
    current_trigger: List[int],
    top_k: int,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    alpha: float = 0.0,
    beta: float = 1.0,
    gamma: float = 0.2,
    delta: float = 0.8,
    zeta: float = 0.0,
    vocab_subset: Optional[List[int]] = None,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    search_width: int = 512,
    n_replace: int = 1,
    layers_to_evaluate: List[int] = None,
    use_gradient_amplification: bool = True,
    nucleus_p: float = 0.9,
    momentum_accumulator: Optional[MomentumGradientAccumulator] = None,
    importance_tracker: Optional[TokenImportanceTracker] = None,
    base_tokens: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Generate candidate trigger sequences using NanoGCG-style gradient sampling.

    Uses raw gradients with topk sampling and supports multi-position replacement.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    device = model.device
    embedding_matrix = model.get_input_embeddings().weight  # (V, d)
    vocab, d = embedding_matrix.shape

    # Allowed ids
    if vocab_subset is None:
        special = set(tokenizer.all_special_ids)
        allowed_ids = torch.tensor(
            [i for i in range(vocab) if i not in special],
            device=device,
            dtype=torch.long,
        )
    else:
        allowed_ids = torch.tensor(
            sorted(set(vocab_subset)), device=device, dtype=torch.long
        )

    E_sub_T = embedding_matrix[allowed_ids].T.contiguous()  # (d, |S|)

    # Trigger param
    ids = torch.tensor(current_trigger, device=device, dtype=torch.long)
    trigger_len = len(current_trigger)

    # Build batched inputs_embeds and masks with the trigger slice tied to a single Parameter
    embeds_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []

    # Parameter shared across prompts
    trigger_embeds_param = torch.nn.Parameter(
        embedding_matrix[ids].detach().clone().unsqueeze(0)  # (1, L, d)
    )

    for prompt in prompts:
        prompt_input_ids = create_prompted_input_ids(
            current_trigger, prompt, tokenizer, n_tk_before
        ).to(
            device
        )  # (1, seq)
        full_input_embeds = (
            embedding_matrix[prompt_input_ids[0]].unsqueeze(0).clone()
        )  # (1, S, d)

        start_pos = n_tk_before
        end_pos = start_pos + trigger_len
        full_input_embeds[:, start_pos:end_pos] = trigger_embeds_param  # tie to param

        embeds_list.append(full_input_embeds.squeeze(0))  # (S, d)
        masks_list.append(
            torch.ones(full_input_embeds.size(1), device=device, dtype=torch.long)
        )  # (S,)

    # Pad to a common length
    max_len = max(e.size(0) for e in embeds_list)
    B = len(embeds_list)
    dtype = embedding_matrix.dtype
    batch_embeds = torch.zeros(B, max_len, d, device=device, dtype=dtype)
    batch_mask = torch.zeros(B, max_len, device=device, dtype=torch.long)
    for i, (e, m) in enumerate(zip(embeds_list, masks_list)):
        L = e.size(0)
        batch_embeds[i, :L, :] = e
        batch_mask[i, :L] = m

    # Mixed precision: enable AMP on CUDA, match model's dtype if half-precision
    use_amp = model.device.type == "cuda"
    amp_dtype = dtype if dtype in [torch.float16, torch.bfloat16] else torch.float16
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        # Define lazy loss functions (only called if coefficient > 0)
        def attention_fn():
            return attention_loss_gradients_batch(
                model=model,
                inputs_embeds=batch_embeds,
                n_tks_trigger=trigger_len,
                n_tk_before=n_tk_before,
                n_tk_after=n_tk_after,
                layers_to_evaluate=layers_to_evaluate,
                alpha=alpha,
                beta=beta,
                reduction="none",
                attention_mask=batch_mask,
            )

        def entropy_fn():
            return entropy_loss_gradients_batch(
                model=model,
                tokenizer=tokenizer,
                inputs_embeds=batch_embeds,
                attention_mask=batch_mask,
                n_tokens=10,  # keep parity with previous default
                maximize_entropy=False,
                reduction="none",
            )

        def divergence_fn():
            # Expand base_tokens for batch: each prompt gets same base tokens
            if base_tokens.size(0) == len(prompts):
                batch_base_tokens = base_tokens  # (num_prompts, n_tokens)
            else:
                batch_base_tokens = base_tokens.expand(B, -1)
            return divergence_loss_gradients_batch(
                model=model,
                tokenizer=tokenizer,
                inputs_embeds=batch_embeds,
                attention_mask=batch_mask,
                base_tokens=batch_base_tokens,
                n_tokens=10,  # keep parity with entropy loss
                temperature=1.0,
                maximize_divergence=True,
                reduction="none",
            )

        combined = compute_combined_loss(
            attention_fn=attention_fn,
            entropy_fn=entropy_fn,
            divergence_fn=divergence_fn if base_tokens is not None else None,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            batch_size=B,
            device=device,
        )

        total_loss = combined.sum()

    # One backward for all prompts
    total_loss.backward()
    grad_embed = trigger_embeds_param.grad.squeeze(0).detach()  # (L, d)

    # Clear the computation graph
    del trigger_embeds_param
    del batch_embeds
    del batch_mask
    del total_loss
    del combined

    # Clear gradients after use
    model.zero_grad(set_to_none=True)

    # Project to subset only
    with torch.no_grad():
        grad_subset = grad_embed @ E_sub_T  # (L, |S|)

        # Momentum adjustment on the projected gradients (subset space)
        if momentum_accumulator is not None:
            grad_subset = momentum_accumulator.update(
                grad_subset=grad_subset,
                subset_indices=allowed_ids,
            )
        # Update token importance tracker
        if importance_tracker is not None:
            importance_tracker.update(
                grad_matrix=grad_subset, subset_indices=allowed_ids
            )

        if use_gradient_amplification and allowed_ids.shape[0] > 1000:
            amplification_temperature = compute_adaptive_temperature(grad_subset)

            print(
                f"Applying gradient amplification (vocab size: {allowed_ids.shape[0]})"
            )

            # Step 1: Temperature scaling to sharpen gradients
            scaled_grads = grad_subset / amplification_temperature

            # Step 2: Convert to probability distribution (negative because we minimize)
            # Use log-softmax for numerical stability
            log_probs = torch.log_softmax(-scaled_grads, dim=1)
            probs = torch.exp(log_probs)

            # Step 3: Nucleus filtering - keep top-p probability mass
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=1)

            # Create mask for tokens within nucleus
            nucleus_mask = cumsum_probs <= nucleus_p
            nucleus_mask[:, 0] = True  # Always keep at least best token

            # Step 4: Create amplified gradient matrix
            amplified_grad_subset = torch.full_like(grad_subset, float("inf"))

            for pos in range(grad_subset.shape[0]):
                # Get indices of tokens in nucleus for this position
                nucleus_indices = sorted_indices[pos][nucleus_mask[pos]]

                # Amplify gradients for nucleus tokens (inverse of probability)
                # This makes high-probability tokens have lower (better) gradients
                amplified_grad_subset[pos, nucleus_indices] = -torch.log(
                    probs[pos, nucleus_indices] + 1e-10
                )

            # Use amplified gradients for candidate selection
            grad_subset = amplified_grad_subset

            # Optional: Print statistics for debugging
            print(
                f"  Nucleus sizes: min={nucleus_mask.sum(dim=1).min().item()}, "
                f"max={nucleus_mask.sum(dim=1).max().item()}, "
                f"mean={nucleus_mask.sum(dim=1).float().mean().item():.1f}"
            )

        # Sample on subset; map back to full ids
    cand_ids_subset = sample_ids_from_grad_rnd_subset(
        ids=ids,
        grad_subset=grad_subset,
        allowed_ids=allowed_ids,
        search_width=min(search_width, top_k),
        topk=top_k,
        n_replace=n_replace,
        temperature=1.0,
        keep_prob=0.1,
    )

    # Clean up
    del grad_embed
    del grad_subset
    del E_sub_T
    torch.cuda.empty_cache()

    return cand_ids_subset


def compute_adaptive_temperature(grad_subset: torch.Tensor) -> float:
    """Compute temperature based on gradient distribution statistics.

    When gradients are very uniform (high entropy), use lower temperature
    to amplify differences. When gradients already have clear winners,
    use higher temperature to avoid over-focusing.
    """
    # Compute entropy of gradient distribution for each position
    grad_probs = torch.softmax(-grad_subset, dim=1)
    entropy = -(grad_probs * torch.log(grad_probs + 1e-10)).sum(dim=1).mean()

    # Normalize entropy to [0, 1] range
    max_entropy = torch.log(torch.tensor(grad_subset.shape[1], dtype=torch.float32))
    normalized_entropy = entropy / max_entropy

    # High entropy (uniform) -> low temperature (more amplification)
    # Low entropy (peaked) -> high temperature (less amplification)
    temperature = 0.01 + 0.99 * (1 - normalized_entropy)

    return temperature.item()


def sample_ids_from_grad_rnd_subset(
    ids: torch.Tensor,
    grad_subset: torch.Tensor,  # (L, |S|)
    allowed_ids: torch.Tensor,  # (|S|,)
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    temperature: float = None,
    keep_prob: float = None,
) -> List[List[int]]:
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    # Get top-k tokens with most negative gradients (steepest descent) in the subset
    k = min(topk, grad_subset.shape[1])
    if k <= 0:
        # No allowed candidates in subset; return originals
        print("no allowed candidates")
        return original_ids.tolist()
    topk_idx = grad_subset.topk(k, dim=1, largest=False).indices  # (L, k)

    # Randomly select positions to replace
    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad_subset.device)
    )[..., :n_replace]

    # Randomly select from top-k tokens for each selected position
    sampled_ids_val_subset = torch.gather(
        topk_idx[sampled_ids_pos],
        2,
        torch.randint(0, k, (search_width, n_replace, 1), device=grad_subset.device),
    ).squeeze(2)

    # Map subset indices back to full vocab IDs
    sampled_ids_val = allowed_ids[sampled_ids_val_subset]

    # Create new candidate sequences
    new_ids = original_ids.scatter(1, sampled_ids_pos, sampled_ids_val)

    return new_ids.tolist()


def gcg_optimization_step(
    current_trigger: List[int],
    best_loss: float,
    prompts: List[str],
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 0.5,
    zeta: float = 0.25,
    top_k: int = 5,
    vocab_subset: Optional[List[int]] = None,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    gradient_based: bool = True,
    n_replace: int = 1,
    layers_to_evaluate: List[int] = None,
    momentum_accumulator: Optional[MomentumGradientAccumulator] = None,
    importance_tracker: Optional[TokenImportanceTracker] = None,
    chunk_size: int = 16,
    base_tokens: Optional[torch.Tensor] = None,
) -> Tuple[List[int], float]:
    """Perform one step of gradient-based GCG optimization.

    Executes a single optimization step using the GCG algorithm to improve a trigger sequence.
    For each position in the current trigger, uses gradients to identify the most promising token substitutions.
    Can also fall back to random candidates if gradient-based generation is not used.
    """
    if gradient_based:
        model.zero_grad(set_to_none=True)

    best_trigger = current_trigger.copy()

    if not gradient_based:
        candidates = generate_token_candidates_random_batched(
            current_trigger=current_trigger,
            top_k=top_k,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layers_to_evaluate=layers_to_evaluate,
            alpha=alpha,
            gamma=gamma,
            vocab_subset=vocab_subset,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            search_width=top_k,
            n_replace=n_replace,
        )
    else:
        candidates = generate_token_candidates(
            current_trigger=current_trigger,
            top_k=top_k,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layers_to_evaluate=layers_to_evaluate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            vocab_subset=vocab_subset,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            search_width=top_k,
            n_replace=n_replace,
            use_gradient_amplification=False,
            momentum_accumulator=momentum_accumulator,
            importance_tracker=importance_tracker,
            base_tokens=base_tokens,
        )

    with torch.no_grad():
        all_losses = []

        for i in range(0, len(candidates), chunk_size):
            chunk_candidates = candidates[i : i + chunk_size]

            chunk_losses = evaluate_multiple_triggers_loss_token_space(
                trigger_tokens_list=chunk_candidates,
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                layers_to_evaluate=layers_to_evaluate,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                zeta=zeta,
                n_tk_before=n_tk_before,
                n_tk_after=n_tk_after,
                base_tokens=base_tokens,
            )
            all_losses.extend(chunk_losses)

            # Clear cache after each chunk
            torch.cuda.empty_cache()

        for candidate, loss in zip(candidates, all_losses):
            if loss < best_loss:
                best_loss = loss
                best_trigger = candidate.copy()

    return best_trigger, best_loss


def run_gradient_based_optimization(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt_strings: List[str],
    trigger_length: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 0.5,
    zeta: float = 0.25,
    num_iterations: int = 20,
    top_k: int = 32,
    vocab_subset: Optional[List[int]] = None,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: List[int] = None,
    gradient_based: bool = True,
    use_backdoor_aware_init: bool = True,  # Add this parameter
    init_n_clusters: int = 100,  # Add this parameter
    # Momentum/adaptive params
    use_momentum: bool = False,
    momentum_type: str = "adam",
    momentum_beta1: float = 0.9,
    momentum_beta2: float = 0.999,
    initial_vocab_size: Optional[int] = None,
    final_vocab_size: Optional[int] = None,
) -> Tuple[List[dict], list]:
    """Run the full GCG optimization algorithm using weighted attention-entropy loss.

    This function initializes a random trigger and iteratively refines it using gradient-based
    candidate generation and weighted attention-entropy loss optimization. It tracks the trigger
    sequence and loss at each step.

    Returns:
        Tuple[List[dict], list]: A tuple containing:
            - A history of the optimization process. Each entry contains:
                - 'iteration': Iteration number.
                - 'loss': Loss value at that iteration.
                - 'trigger_tokens': Current trigger token IDs.
                - 'trigger_string': Current trigger as a decoded string.
            - A list of the best trigger tokens found during optimization.

    """
    print("=" * 80)
    print(f"GRADIENT-BASED OPTIMIZATION")
    print("=" * 80)
    print("Initializing new trigger for gradient-based optimization...")
    # Momentum accumulator and importance tracker
    mom_accum: Optional[MomentumGradientAccumulator] = None
    importance_tracker: Optional[TokenImportanceTracker] = None
    if use_momentum and gradient_based:
        vocab_size = model.get_input_embeddings().weight.shape[0]
        mom_accum = MomentumGradientAccumulator(
            trigger_length=trigger_length,
            vocab_size=vocab_size,
            momentum_type=momentum_type,
            beta1=momentum_beta1,
            beta2=momentum_beta2,
            device=model.device,
            dtype=torch.float32,
        )
        importance_tracker = TokenImportanceTracker(
            vocab_size=vocab_size,
            trigger_length=trigger_length,
            device=model.device,
            dtype=torch.float32,
        )
    if use_backdoor_aware_init:
        gradient_trigger = initialize_backdoor_aware_trigger(
            length=trigger_length,
            tokenizer=tokenizer,
            model=model,
            vocab_subset=vocab_subset,
            prompts=prompt_strings,
            n_clusters=init_n_clusters,
        )
    else:
        gradient_trigger = initialize_random_trigger(
            trigger_length,
            tokenizer,
            vocab_subset,
        )
    print(f"Initial gradient trigger tokens: {gradient_trigger}")
    print(f"Initial trigger string: '{tokens_to_string(gradient_trigger, tokenizer)}'")

    # Generate base tokens cache if divergence loss is used (zeta > 0)
    base_tokens_cache = None
    if zeta > 0.0:
        print("Generating base tokens cache for divergence loss...")
        base_cache_dict = generate_base_tokens(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_strings,
            n_tokens=10,  # Keep parity with entropy loss
            use_chat_template=True,
        )
        base_tokens_cache = base_cache_dict["base_tokens"].to(model.device)
        print(f"Base tokens cache shape: {base_tokens_cache.shape}")

    # Evaluate initial loss
    with torch.no_grad():
        gradient_initial_loss = evaluate_multiple_triggers_loss_token_space(
            trigger_tokens_list=[gradient_trigger],
            prompts=prompt_strings,
            model=model,
            tokenizer=tokenizer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            layers_to_evaluate=layers_to_evaluate,
            base_tokens=base_tokens_cache,
        )[0]
    print(f"Initial gradient loss: {gradient_initial_loss:.6f}")

    # Store optimization history
    gradient_history = []
    gradient_best_trigger = gradient_trigger.copy()
    gradient_best_loss = gradient_initial_loss

    print(f"Running gradient-based optimization for {num_iterations} iterations...")
    if vocab_subset is not None:
        print(f"Using restricted vocabulary subset of size {len(vocab_subset)}")
    print("=" * 80)

    n_replace = trigger_length
    iterations_since_improvement = 0
    # Optional adaptive subset size schedule
    scheduler: Optional[AdaptiveSparsityScheduler] = None
    if vocab_subset is not None and initial_vocab_size and final_vocab_size:
        initial_vocab_size = min(len(vocab_subset), int(initial_vocab_size))
        final_vocab_size = min(len(vocab_subset), int(final_vocab_size))
        scheduler = AdaptiveSparsityScheduler(
            initial_vocab_size=initial_vocab_size,
            final_vocab_size=final_vocab_size,
            total_iterations=num_iterations,
            transition_type="cosine",
            warmup_steps=max(1, num_iterations // 10),
        )
    pbar = tqdm(range(num_iterations), dynamic_ncols=True)
    pbar.set_description(
        f"Gradient Optimization | best: {gradient_best_loss:.6f} | replace: {n_replace}"
    )

    for iteration in pbar:
        current_subset = vocab_subset
        # Adapt subset during training
        if scheduler is not None:
            current_size = scheduler.step()
            # Build subset that preserves mandatory tokens (e.g., original best trigger tokens)
            mandatory = []
            if gradient_best_trigger:
                mandatory = list(set(gradient_best_trigger))

            # Get important tokens from gradient history
            if importance_tracker is not None and iteration > 0:
                # Get tokens that have shown promise across iterations
                num_important = max(0, current_size - len(mandatory))
                if num_important > 0:
                    important_tokens = importance_tracker.top_tokens(
                        num_tokens=num_important
                    ).tolist()

                    # Add to mandatory tokens (avoiding duplicates)
                    mandatory.extend(
                        [tok for tok in important_tokens if tok not in mandatory]
                    )

            current_subset = build_vocab_subset(
                tokenizer=tokenizer,
                vocab_size=model.get_input_embeddings().weight.shape[0],
                mandatory_tokens=mandatory,
                subset_size=int(current_size),
            )

        # Perform one gradient-based optimization step
        gradient_trigger, gradient_current_loss = gcg_optimization_step(
            current_trigger=gradient_trigger,
            best_loss=gradient_best_loss,
            prompts=prompt_strings,
            model=model,
            tokenizer=tokenizer,
            layers_to_evaluate=layers_to_evaluate,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            top_k=top_k,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            vocab_subset=current_subset,
            n_replace=n_replace,
            gradient_based=gradient_based,
            momentum_accumulator=mom_accum,
            importance_tracker=importance_tracker,
            base_tokens=base_tokens_cache,
        )

        # Update best if improved
        trsh = math.ceil(num_iterations * 0.05)
        if iterations_since_improvement == trsh and n_replace > 1:
            n_replace = max(1, int(n_replace * 0.8))
            iterations_since_improvement = 0
        if (
            gradient_current_loss <= gradient_best_loss
            and gradient_trigger != gradient_best_trigger
        ):
            gradient_best_loss = gradient_current_loss
            gradient_best_trigger = gradient_trigger.copy()
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        # Store history
        gradient_history.append(
            {
                "iteration": iteration,
                "loss": gradient_current_loss,
                "trigger_tokens": gradient_trigger.copy(),
                "trigger_string": tokens_to_string(gradient_trigger, tokenizer),
                "n_replace": n_replace,
                "subset_size": (
                    len(current_subset) if current_subset is not None else None
                ),
            }
        )

        # Print progress
        if iteration % 5 == 0 or iteration == num_iterations - 1:
            print(
                f"Gradient Iter {iteration + 1:2d}: Loss = {gradient_current_loss:.6f}, "
                f"Trigger = '{tokens_to_string(gradient_trigger, tokenizer)}'"
            )

        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        # Stop if no improvement for too long
        if iterations_since_improvement >= 15 and n_replace == 1:
            print(f"STOP: No improvement for {iterations_since_improvement} rounds.")
            break

        # # Reduce n_replace over time: 20% fewer tokens every 10% of iterations - min 1
        # if (iteration + 1) % (num_iterations // 10) == 0:
        #     n_replace = max(1, int(n_replace * 0.8))

        pbar.set_description(
            f"Dev: {model.device} | best: {gradient_best_loss:.6f} | r: {n_replace} | n-i: {iterations_since_improvement} | s: {len(current_subset)}"
        )

    print("=" * 80)
    print(f"Best gradient loss: {gradient_best_loss:.6f}")
    print(
        f"Best gradient trigger: '{tokens_to_string(gradient_best_trigger, tokenizer)}'"
    )
    print(
        f"Gradient loss improvement: {gradient_initial_loss - gradient_best_loss:.6f} "
        f"({((gradient_initial_loss - gradient_best_loss) / abs(gradient_initial_loss) * 100):.2f}%)"
    )

    return gradient_history, gradient_best_trigger
