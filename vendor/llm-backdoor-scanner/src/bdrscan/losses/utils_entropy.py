# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
from torch import Tensor
from typing import Optional
from transformers import PreTrainedTokenizer
from bdrscan.losses.utils_rollout import next_token_logits


def compute_entropy(logits: Tensor) -> Tensor:
    """
    Compute the entropy of the probability distribution over logits.
    """
    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def _get_initial_embeds_and_mask(
    model: torch.nn.Module,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    attention_mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Returns:
      embeds: (B, L, D)
      attention_mask: (B, L)
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided.")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide only one of input_ids or inputs_embeds.")

    if inputs_embeds is not None:
        embeds = inputs_embeds
        if embeds.dim() == 2:  # (L, D) -> (1, L, D)
            embeds = embeds.unsqueeze(0)
    else:
        # Normalize input_ids to (B, L) then embed
        if input_ids.dim() == 1:  # (L,) -> (1, L)
            input_ids = input_ids.unsqueeze(0)
        embeds = model.get_input_embeddings()(input_ids.to(model.device))

    embeds = embeds.to(model.device)
    if attention_mask is None:
        attention_mask = torch.ones(embeds.shape[:2], device=embeds.device)
    else:
        attention_mask = attention_mask.to(embeds.device)
    return embeds, attention_mask


def _rollout_expected_embeddings(
    model: torch.nn.Module,
    embeds: Tensor,  # (B, L, D)
    attention_mask: Tensor,  # (B, L)
    n_tokens: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Iteratively:
      - forward pass to get logits
      - compute entropy and probs for the next token
      - append expected embedding (probs @ embedding_matrix) for the next step

    Returns lists of length n_tokens:
      entropies_per_step: list of (B,)
      probs_per_step:     list of (B, V)
    """
    if n_tokens <= 0:
        raise ValueError("n_tokens must be > 0")

    B = embeds.size(0)
    _ = B  # silence linters
    embedding_matrix = model.get_input_embeddings().weight  # (V, D)

    entropies_per_step: list[Tensor] = []
    probs_per_step: list[Tensor] = []

    for step in range(n_tokens):
        logits = next_token_logits(model, embeds, attention_mask)

        step_entropy = compute_entropy(logits)  # (B,)
        entropies_per_step.append(step_entropy)

        probs = torch.softmax(logits.float(), dim=-1)  # (B, V)
        probs_per_step.append(probs)

        if step < n_tokens - 1:
            exp_embed = probs.to(embedding_matrix.dtype) @ embedding_matrix  # (B, D)
            embeds = torch.cat([embeds, exp_embed.unsqueeze(1)], dim=1)  # (B, L+1, D)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((embeds.size(0), 1), device=attention_mask.device),
                ],
                dim=1,
            )

    return entropies_per_step, probs_per_step


def _penalties_from_probs(
    probs_per_step: list[Tensor],  # list of (B, V)
    eos_id: Optional[int],
    early_termination_penalty: float,
    repetition_penalty: float,
    use_torch_log_for_max: bool = False,
) -> Tensor:
    """
    Vectorized penalties used by both single/batch variants.
    Compute entropy directly from average probabilities.
    Supports the two equivalent max-entropy normalizations:
      - single-seq historically used torch.log(tensor(...))
      - batched variants used math.log(...)
    """
    if not probs_per_step:
        raise ValueError("probs_per_step must be non-empty")

    B = probs_per_step[0].size(0)
    device = probs_per_step[0].device
    penalties = torch.zeros(B, device=device)

    # Early EOS penalties
    if eos_id is not None:
        penalties = penalties + early_termination_penalty * probs_per_step[0][:, eos_id]
        if len(probs_per_step) >= 2:
            penalties = (
                penalties
                + early_termination_penalty * 0.5 * probs_per_step[1][:, eos_id]
            )

    # Repetition penalty via entropy of average distribution
    avg_probs = torch.stack(probs_per_step, dim=0).float().mean(dim=0)
    log_probs = avg_probs.clamp_min(torch.finfo(avg_probs.dtype).tiny).log()
    avg_entropy = -(avg_probs * log_probs).sum(dim=-1)

    if use_torch_log_for_max:
        max_entropy = torch.log(
            torch.tensor(
                avg_probs.size(-1), dtype=avg_probs.dtype, device=avg_probs.device
            )
        )
    else:
        max_entropy = math.log(avg_probs.size(-1))

    rep_penalty = 1.0 - (avg_entropy / max_entropy)  # (B,)
    penalties = penalties + repetition_penalty * rep_penalty

    return penalties


# ------------------------------- LOSS


def entropy_loss_no_grad_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    n_tokens: int = 5,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
    maximize_entropy: bool = False,
    reduction: str = "none",
) -> Tensor:
    """
    Batched no-grad version of gcg_entropy_loss_with_gradients using full
    recomputation each step. Mirrors the single-sequence logic:
    - Iteratively appends expected embedding (probs * embedding_matrix)
    - Averages entropies over the n generated steps
    - Adds early EOS + repetition penalties
    - Optional maximize_entropy sign flip

    Args mirror the original plus:
        reduction: 'mean' returns scalar; 'none' returns per-sample tensor (B,).

    Returns:
        Tensor: Scalar (if reduction='mean') or (B,) losses.
    """

    with torch.no_grad():
        embeds, attention_mask = _get_initial_embeds_and_mask(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        embeds = embeds.to(model.device)
        entropies_per_step, probs_per_step = _rollout_expected_embeddings(
            model, embeds, attention_mask, n_tokens=n_tokens
        )

        entropy_loss = torch.stack(entropies_per_step, dim=0).mean(dim=0)  # (B,)

        penalties = _penalties_from_probs(
            probs_per_step=probs_per_step,
            eos_id=tokenizer.eos_token_id,
            early_termination_penalty=early_termination_penalty,
            repetition_penalty=repetition_penalty,
            use_torch_log_for_max=False,  # match original batched path
        )

        total = entropy_loss + penalties
        if maximize_entropy:
            total = -total

        if reduction == "mean":
            return total.mean()
        if reduction == "none":
            return total
        raise ValueError("reduction must be 'mean' or 'none'")


def entropy_loss_gradients_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    n_tokens: int = 5,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
    maximize_entropy: bool = False,
    reduction: str = "none",
) -> Tensor:
    """
    Batched, fully differentiable version of `entropy_loss_with_gradients`.
    Mirrors the single-sequence logic across a batch:
      - Iteratively appends expected embedding (probs @ embedding_matrix)
      - Averages entropies over the n generated steps
      - Adds early EOS + repetition penalties
      - Optional maximize_entropy sign flip
    Gradients flow through model parameters and through `inputs_embeds` if it requires_grad.

    Returns:
        Tensor: Scalar if reduction='mean', else per-sample (B,)
    """
    embeds, attention_mask = _get_initial_embeds_and_mask(
        model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    entropies_per_step, probs_per_step = _rollout_expected_embeddings(
        model, embeds, attention_mask, n_tokens=n_tokens
    )

    # Average entropy across generated steps (per sample)
    entropy_loss = torch.stack(entropies_per_step, dim=0).mean(dim=0)  # (B,)

    penalties = _penalties_from_probs(
        probs_per_step=probs_per_step,
        eos_id=tokenizer.eos_token_id,
        early_termination_penalty=early_termination_penalty,
        repetition_penalty=repetition_penalty,
        use_torch_log_for_max=False,  # match original batched path
    )

    total = entropy_loss + penalties

    if maximize_entropy:
        total = -total

    if reduction == "mean":
        return total.mean()
    if reduction == "none":
        return total
    raise ValueError("reduction must be 'mean' or 'none'")


# ---------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    print("This module is not intended to be run directly.")
    print("Use it as part of a larger training or evaluation script.")
    print("This test only ensures gradients flows through the loss function.")
    from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
    from bdrscan.utils_model import load_model_and_tokenizer
    from bdrscan.losses.old_single_input_losses import entropy_loss_with_gradients

    import time
    import gc
    import numpy as np

    def benchmark(fn, *args, warmup=1, runs=5, label="", **kwargs):
        # Warmup
        for _ in range(warmup):
            _ = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        avg = dt / runs
        print(f"{label:<35} {avg*1000:8.2f} ms / call (avg over {runs})")
        return avg

    def gradient_single_batch_runner(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        batch_ids: torch.Tensor,
        n_tokens: int,
    ):
        losses = []
        # Loop over samples; run forward+backward to include gradient cost.
        for i in range(batch_ids.size(0)):
            ids = batch_ids[i]
            if ids.dim() == 1:  # ensure (1, L) for embedding lookup inside loss
                ids = ids.unsqueeze(0)
            # Ensure 1D input_ids
            loss = entropy_loss_with_gradients(
                model=model,
                tokenizer=tokenizer,
                input_ids=ids,
                n_tokens=n_tokens,
            )
            model.zero_grad(set_to_none=True)
            losses.append(loss)

        return torch.stack(losses)

    model, tokenizer, config = load_model_and_tokenizer(
        model_name="llama",
        cache_dir=CACHE_DIR,
        config=MODELS_CONFIG,
        dtype="float16",
        device=f"cuda:1",
    )

    print("\nGRADIENT BASED VERSION")
    input_embeds = torch.randn(
        1, 10, 4096, requires_grad=True, dtype=torch.float16, device=model.device
    )
    loss = entropy_loss_with_gradients(
        model=model, tokenizer=tokenizer, inputs_embeds=input_embeds, n_tokens=5
    )
    loss.backward()
    print(input_embeds.grad)

    del input_embeds, loss

    print("\nBATCHED VERSION")

    sample_ids = torch.randint(low=100, high=5000, size=(4, 12), device=model.device)
    batched_losses = entropy_loss_no_grad_batch(
        model=model,
        tokenizer=tokenizer,
        input_ids=sample_ids,
        n_tokens=10,
        reduction="none",
    )
    single_losses = gradient_single_batch_runner(
        model=model,
        tokenizer=tokenizer,
        batch_ids=sample_ids,
        n_tokens=10,
    )
    # ---- Batched WITH gradients: parity vs single-seq loop ----
    print("\nBATCHED WITH GRADIENTS")
    batched_grad_losses = entropy_loss_gradients_batch(
        model=model,
        tokenizer=tokenizer,
        input_ids=sample_ids,
        n_tokens=10,
        reduction="none",
    )

    print("Batched losses     :", batched_losses.tolist())
    print("Single losses      :", single_losses.tolist())
    print("Batched grad losses:", batched_grad_losses.tolist())
    print(
        "batch:",
        batched_losses.mean().item(),
        "single:",
        single_losses.mean().item(),
        "batched grad:",
        batched_grad_losses.mean().item(),
    )

    # Quick gradient-flow smoke test on batched embeds
    emb_dim = model.get_input_embeddings().weight.size(1)
    test_embeds = torch.randn(
        3, 7, emb_dim, device=model.device, dtype=model.dtype, requires_grad=True
    )
    loss_bmean = entropy_loss_gradients_batch(
        model=model,
        tokenizer=tokenizer,
        inputs_embeds=test_embeds,
        n_tokens=5,
        reduction="mean",
    )
    loss_bmean.backward()
    grad_norm = test_embeds.grad.norm().item()
    print(f"Gradient norm (batched inputs_embeds): {grad_norm:.6f}")

    del (
        batched_losses,
        single_losses,
        sample_ids,
        batched_grad_losses,
        grad_norm,
        test_embeds,
    )
    gc.collect()
    torch.cuda.empty_cache()

    large_ids = torch.randint(low=100, high=5000, size=(16, 24), device=model.device)
    print("\nTiming (batch size=16, seq_len=24, n_tokens=3)")
    t_grad_large = benchmark(
        gradient_single_batch_runner,
        model,
        tokenizer,
        large_ids,
        n_tokens=10,
        label="Gradient (loop, backward)",
        runs=2,
        warmup=1,
    )
    t_plain_large = benchmark(
        entropy_loss_no_grad_batch,
        model,
        tokenizer,
        large_ids,
        n_tokens=10,
        reduction="mean",
        label="Plain batched (recompute)",
    )
    t_grad_batch_large = benchmark(
        entropy_loss_gradients_batch,
        model,
        tokenizer,
        large_ids,
        n_tokens=10,
        reduction="mean",
        label="Batched with gradients (forward)",
    )
    print(f"Avg time gradient: {t_grad_large*1000:0.2f} ms / call")
    print(f"Avg time plain: {t_plain_large*1000:0.2f} ms / call")
    print(f"Avg time batched grad: {t_grad_batch_large*1000:0.2f} ms / call")
    print(f"Speedup vs gradient: plain {t_grad_large / t_plain_large:0.2f}x")
    print(
        f"Speedup vs gradient: batched gradients {t_grad_large / t_grad_batch_large:0.2f}x"
    )
    print(
        f"Slowdown vs plain batched: batched gradients {t_grad_batch_large / t_plain_large:0.2f}x"
    )
