# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch import Tensor
from typing import List, Optional


# -------------------------- HELPERS


def _resolve_layers_to_evaluate(
    layers_to_evaluate: Optional[List[int]],
    total_layers: int,
) -> List[int]:
    """
    Derive layers_to_evaluate default as the middle +/- 4, then clip to valid indices.
    total_layers is sourced from:
      - model.config.num_hidden_layers for the single-input function (parity),
      - len(attentions) for batched functions (parity).
    """
    if layers_to_evaluate == -1:
        layers_to_evaluate = list(range(total_layers))
    elif layers_to_evaluate is None:
        mid = total_layers // 2
        layers_to_evaluate = list(range(mid - 4, mid + 4))
    # Clip to valid range
    layers = [i for i in layers_to_evaluate if 0 <= i < total_layers]
    return layers


def _aggregate_attn(attentions: List[Tensor], layers: List[int]) -> Tensor:
    """
    Stack selected layers and average across layers and heads -> (B, S, S)
    attentions: list of tensors each (B,H,S,S)
    """
    selected = [attentions[i] for i in layers if attentions[i] is not None]
    if not selected:
        raise ValueError("No dense attention matrices available for ranking; set gamma=0 to explicitly use only entropy/divergence.")
    for value in selected:
        if value.ndim != 4 or value.shape[-1] != value.shape[-2]:
            raise ValueError("Attention ranking requires (batch, heads, sequence, sequence) matrices")
    # Models may return different head counts for different layers.
    return torch.stack([value.mean(dim=1) for value in selected]).mean(dim=0)


def _forward_attentions(
    model: torch.nn.Module,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    attention_mask: Optional[Tensor],
):
    """
    Normalizes inputs to batch-first form and runs the model with output_attentions=True.
    Returns (attentions_list, attention_mask) where attentions_list is List[Tensor] each (B,H,S,S),
    and attention_mask is passed through on the correct device (shape (B,L) if provided).
    """
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide only one of input_ids or inputs_embeds")

    kwargs = dict(output_attentions=True, return_dict=True)

    if input_ids is not None:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    else:
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        inputs_embeds = inputs_embeds.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        outputs = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    raw = getattr(outputs, "attentions", None)
    matrices = [value for value in (raw or ()) if value is not None]
    if not matrices:
        raise ValueError("Model returned no dense attention matrices. Use eager attention or explicitly set gamma=0.")
    config = getattr(model.config, "text_config", model.config)
    types = getattr(config, "layer_types", None)
    full_layers = [i for i, kind in enumerate(types or []) if kind == "full_attention"]
    if types and len(raw) == len(types):
        indices = [i for i, value in enumerate(raw) if value is not None]
    elif full_layers and len(matrices) == len(full_layers):
        indices = full_layers
    elif not types and len(matrices) == getattr(config, "num_hidden_layers", None):
        indices = list(range(len(matrices)))
    else:
        indices = None
    model.bdrscan_attention_coverage = {
        "total_model_layers": getattr(config, "num_hidden_layers", None),
        "returned_dense_matrices": len(matrices), "dense_layer_indices": indices,
        "scope": "returned_dense_attention_only",
        "selection": "middle_up_to_eight_returned_matrices",
    }
    return matrices, attention_mask


def _block_mean_strict_lower(mat: Tensor) -> Tensor:
    T = mat.size(0)
    if T <= 1:
        return mat.sum() * 0.0
    mask = torch.tril(
        torch.ones((T, T), device=mat.device, dtype=torch.bool), diagonal=-1
    )
    return mat.masked_select(mask).mean()


def _block_mean_offdiag(mat: Tensor) -> Tensor:
    T = mat.size(0)
    if T <= 1:
        return mat.sum() * 0.0
    eye = torch.eye(T, device=mat.device, dtype=torch.bool)
    return mat.masked_select(~eye).mean()


def _compute_per_sample_attention_loss(
    attn_mean: Tensor,  # (B, S, S)
    attention_mask: Optional[Tensor],
    n_tks_trigger: int,
    n_tk_before: int,
    n_tk_after: int,
    alpha: float,
    beta: float,
    differentiable_zero: bool,
) -> Tensor:
    """
    Implements the per-sample trimming by attention_mask, template exclusion, and loss terms.
    Returns (B,) tensor of losses. Matches the original logic for both batched variants.
    """
    B, S, _ = attn_mean.shape
    dtype = attn_mean.dtype
    device = attn_mean.device

    per_sample = torch.zeros(B, device=device, dtype=dtype)

    for b in range(B):
        attn_sample = attn_mean[b]  # (S, S)

        # Trim to contiguous active region if attention_mask provided
        if attention_mask is not None:
            mask_b = attention_mask[b].to(dtype=torch.bool)
            idx = torch.nonzero(mask_b, as_tuple=False).flatten()
            if idx.numel() == 0:
                zero = (
                    attn_sample.sum() * 0.0
                    if differentiable_zero
                    else torch.zeros((), device=device, dtype=dtype)
                )
                per_sample[b] = zero
                continue
            start = int(idx[0].item())
            end = int(idx[-1].item()) + 1
            attn_trim = attn_sample[start:end, start:end]
            seq_len = end - start
        else:
            attn_trim = attn_sample
            seq_len = attn_trim.shape[0]

        # Apply template exclusion symmetrically
        if n_tk_after > 0:
            if seq_len > n_tk_before + n_tk_after:
                attn_excluded = attn_trim[
                    n_tk_before : seq_len - n_tk_after,
                    n_tk_before : seq_len - n_tk_after,
                ]
            else:
                zero = (
                    attn_trim.sum() * 0.0
                    if differentiable_zero
                    else torch.zeros((), device=device, dtype=dtype)
                )
                per_sample[b] = zero
                continue
        else:
            if seq_len > n_tk_before:
                attn_excluded = attn_trim[n_tk_before:seq_len, n_tk_before:seq_len]
            else:
                zero = (
                    attn_trim.sum() * 0.0
                    if differentiable_zero
                    else torch.zeros((), device=device, dtype=dtype)
                )
                per_sample[b] = zero
                continue

        S_eff = attn_excluded.shape[0]
        zero = (
            attn_excluded.sum() * 0.0
            if differentiable_zero
            else torch.zeros((), device=device, dtype=dtype)
        )

        # Trigger -> Trigger maximization (negative mean over T x T)
        if alpha != 0.0 and 0 < n_tks_trigger <= S_eff:
            trigger_block = attn_excluded[:n_tks_trigger, :n_tks_trigger]
            # Use off-diagonal mean to exclude self-attention (diagonal)
            # This focuses on cross-token attention within the trigger
            trig_mean = _block_mean_offdiag(trigger_block)
            trigger_term = -trig_mean * alpha
        else:
            trigger_term = zero

        # Prompt -> Trigger minimization (mean over (S_eff - T) x T)
        if 0 < n_tks_trigger < S_eff:
            prompt_to_trigger_block = attn_excluded[n_tks_trigger:, :n_tks_trigger]
            prompt_term = prompt_to_trigger_block.mean() * beta
        else:
            prompt_term = zero

        per_sample[b] = trigger_term + prompt_term

    return per_sample


# --------------------------- LOSS


def attention_loss_no_grad_batch(
    model: torch.nn.Module,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    n_tks_trigger: int = 0,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: Optional[List[int]] = None,
    alpha: float = 0.0,
    beta: float = 1.0,
    reduction: str = "none",
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Batched no-grad variant of gcg_attention_loss_with_gradients.

    Mirrors the single-input logic numerically:
    - Maximize trigger→trigger attention: -alpha * mean(attn[:T, :T])
    - Minimize prompt→trigger attention: +beta * mean(attn[T:, :T])
    with the same layer/head averaging and template exclusion.
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide only one of input_ids or inputs_embeds")

    with torch.no_grad():
        attentions, attention_mask = _forward_attentions(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        num_layers_total = len(attentions)
        layers = _resolve_layers_to_evaluate(layers_to_evaluate, num_layers_total)

        attn_mean = _aggregate_attn(attentions, layers)

        per_sample = _compute_per_sample_attention_loss(
            attn_mean=attn_mean,
            attention_mask=attention_mask,
            n_tks_trigger=n_tks_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            alpha=alpha,
            beta=beta,
            differentiable_zero=False,
        )

        if reduction == "mean":
            return per_sample.mean()
        elif reduction == "none":
            return per_sample
        else:
            raise ValueError("reduction must be 'mean' or 'none'")


def attention_loss_gradients_batch(
    model: torch.nn.Module,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    n_tks_trigger: int = 0,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: Optional[List[int]] = None,
    alpha: float = 0.0,
    beta: float = 1.0,
    reduction: str = "none",
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Batched, fully differentiable variant of attention_loss_with_gradients.

    Mirrors the single-input behavior per sample:
    - Average attentions over selected layers and heads -> (B, S, S)
    - Exclude template tokens [n_tk_before : -n_tk_after] symmetrically
    - Maximize trigger→trigger:       -alpha * mean(attn[:T, :T])
    - Minimize prompt→trigger:        +beta  * mean(attn[T:, :T])

    Returns:
        Tensor: (B,) if reduction='none', else scalar mean.
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide only one of input_ids or inputs_embeds")

    attentions, attention_mask = _forward_attentions(
        model=model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )

    num_layers_total = len(attentions)
    layers = _resolve_layers_to_evaluate(layers_to_evaluate, num_layers_total)

    attn_mean = _aggregate_attn(attentions, layers)

    per_sample = _compute_per_sample_attention_loss(
        attn_mean=attn_mean,
        attention_mask=attention_mask,
        n_tks_trigger=n_tks_trigger,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        alpha=alpha,
        beta=beta,
        differentiable_zero=True,
    )

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    raise ValueError("reduction must be 'mean' or 'none'")


# ------------------------------------------------


def _mean_strict_lower(mat: Tensor) -> Tensor:
    T = mat.size(0)
    if T <= 1:
        return mat.sum() * 0.0
    mask = torch.tril(
        torch.ones((T, T), dtype=torch.bool, device=mat.device), diagonal=-1
    )
    denom = mask.sum().clamp_min(1)
    return mat.masked_select(mask).sum() / denom


def _mean_offdiag(mat: Tensor) -> Tensor:
    T = mat.size(0)
    if T <= 1:
        return mat.sum() * 0.0
    eye = torch.eye(T, dtype=torch.bool, device=mat.device)
    mask = ~eye
    denom = mask.sum().clamp_min(1)
    return mat.masked_select(mask).sum() / denom


def attention_sanity_checks_batch(
    model: torch.nn.Module,
    *,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    n_tks_trigger: int,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: Optional[List[int]] = None,
    last_k_prompt: int = 32,
) -> dict[str, Tensor]:
    """
    Compute diagnostics that explain the attention loss behavior.

    Returns a dict of (B,) tensors:
      - trigger_diag_mean
      - trigger_strict_lower_mean
      - trigger_offdiag_mean
      - prompt_to_trigger_mean
      - prompt_to_trigger_lastk_mean

    Notes:
      - Uses the same layer averaging and template exclusion as the loss.
      - Strict-lower uses only i>j in the trigger block (causal cross-token cohesion).
      - last_k_prompt is taken from the end of the prompt region (rows T:).
    """
    # Forward and aggregate attentions
    attentions, attn_mask = _forward_attentions(
        model=model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    total_layers = len(attentions)
    layers = _resolve_layers_to_evaluate(layers_to_evaluate, total_layers)
    attn_mean = _aggregate_attn(attentions, layers)  # (B,S,S)

    B, S, _ = attn_mean.shape
    device = attn_mean.device
    dtype = attn_mean.dtype

    def zero_like() -> Tensor:
        return torch.zeros((), device=device, dtype=dtype)

    # Prepare outputs
    trig_diag = torch.zeros(B, device=device, dtype=dtype)
    trig_lower = torch.zeros(B, device=device, dtype=dtype)
    trig_offdiag = torch.zeros(B, device=device, dtype=dtype)
    p2t_mean = torch.zeros(B, device=device, dtype=dtype)
    p2t_lastk = torch.zeros(B, device=device, dtype=dtype)

    for b in range(B):
        A = attn_mean[b]  # (S,S)

        # Trim contiguous active region by attention_mask if provided
        if attn_mask is not None:
            mask_b = attn_mask[b].to(torch.bool)
            idx = torch.nonzero(mask_b, as_tuple=False).flatten()
            if idx.numel() == 0:
                trig_diag[b] = zero_like()
                trig_lower[b] = zero_like()
                trig_offdiag[b] = zero_like()
                p2t_mean[b] = zero_like()
                p2t_lastk[b] = zero_like()
                continue
            start = int(idx[0])
            end = int(idx[-1]) + 1
            Atrim = A[start:end, start:end]
            Seff = end - start
        else:
            Atrim = A
            Seff = Atrim.size(0)

        # Exclude template region [n_tk_before : -n_tk_after]
        if n_tk_after > 0:
            if Seff <= n_tk_before + n_tk_after:
                trig_diag[b] = Atrim.sum() * 0.0
                trig_lower[b] = Atrim.sum() * 0.0
                trig_offdiag[b] = Atrim.sum() * 0.0
                p2t_mean[b] = Atrim.sum() * 0.0
                p2t_lastk[b] = Atrim.sum() * 0.0
                continue
            Aex = Atrim[
                n_tk_before : Seff - n_tk_after, n_tk_before : Seff - n_tk_after
            ]
        else:
            if Seff <= n_tk_before:
                trig_diag[b] = Atrim.sum() * 0.0
                trig_lower[b] = Atrim.sum() * 0.0
                trig_offdiag[b] = Atrim.sum() * 0.0
                p2t_mean[b] = Atrim.sum() * 0.0
                p2t_lastk[b] = Atrim.sum() * 0.0
                continue
            Aex = Atrim[n_tk_before:Seff, n_tk_before:Seff]

        Seff2 = Aex.size(0)
        T = n_tks_trigger

        # Trigger-block stats
        if 0 < T <= Seff2:
            Tblock = Aex[:T, :T]
            trig_diag[b] = Tblock.diagonal().mean()
            trig_lower[b] = _mean_strict_lower(Tblock)
            trig_offdiag[b] = _mean_offdiag(Tblock)
        else:
            z = Aex.sum() * 0.0
            trig_diag[b] = z
            trig_lower[b] = z
            trig_offdiag[b] = z

        # Prompt -> Trigger stats
        if 0 < T < Seff2:
            P2T = Aex[T:, :T]
            p2t_mean[b] = P2T.mean()
            if P2T.size(0) > 0:
                k = min(last_k_prompt, P2T.size(0))
                p2t_lastk[b] = P2T[-k:, :].mean()
            else:
                p2t_lastk[b] = P2T.sum() * 0.0
        else:
            z = Aex.sum() * 0.0
            p2t_mean[b] = z
            p2t_lastk[b] = z

    return {
        "trigger_diag_mean": trig_diag,  # (B,)
        "trigger_strict_lower_mean": trig_lower,  # (B,)
        "trigger_offdiag_mean": trig_offdiag,  # (B,)
        "prompt_to_trigger_mean": p2t_mean,  # (B,)
        "prompt_to_trigger_lastk_mean": p2t_lastk,  # (B,)
    }


# -------------------------------------------------


if __name__ == "__main__":
    # Example usage
    print("This module is not intended to be run directly.")
    print("Use it as part of a larger training or evaluation script.")
    print("This test only esnures gradients flows through the loss function.")
    from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
    from bdrscan.losses.old_single_input_losses import attention_loss_with_gradients
    from bdrscan.utils_model import (
        load_model_and_tokenizer,
        count_chat_template_tokens,
    )
    import gc
    import time

    def benchmark(fn, *args, warmup=1, runs=5, label="", **kwargs):
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
        n_tks_trigger: int,
        n_tk_before: int,
        n_tk_after: int,
        layers_to_evaluate,
        alpha: float,
    ):
        # Per-sample forward+backward to reflect gradient cost
        for i in range(batch_ids.size(0)):
            ids = batch_ids[i]
            _ = attention_loss_with_gradients(
                model=model,
                input_ids=ids.unsqueeze(0),
                n_tks_trigger=n_tks_trigger,
                n_tk_before=n_tk_before,
                n_tk_after=n_tk_after,
                layers_to_evaluate=layers_to_evaluate,
                alpha=alpha,
            )
            model.zero_grad(set_to_none=True)

    model, tokenizer, config = load_model_and_tokenizer(
        model_name="llama",
        cache_dir=CACHE_DIR,
        config=MODELS_CONFIG,
        dtype="float16",
        device=f"cuda",
    )
    n_tk_before, n_tk_trigger, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt="test trigger"
    )

    print("\nSINGLE INPUT VERSION")
    input_embeds = torch.randn(
        1, 10, 4096, requires_grad=True, dtype=torch.float16, device=model.device
    )
    loss = attention_loss_with_gradients(
        model=model,
        inputs_embeds=input_embeds,
        n_tks_trigger=n_tk_trigger,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
    )
    loss.backward()
    print(input_embeds.grad)

    del input_embeds, loss
    gc.collect()
    torch.cuda.empty_cache()

    prompt_ids = torch.randint(low=100, high=4000, size=(4, 32), device=model.device)
    n_before, n_trigger, n_after = 0, 5, 0

    # Single (loop) reference
    ref = []
    for i in range(prompt_ids.size(0)):
        loss_i = attention_loss_with_gradients(
            model=model,
            input_ids=prompt_ids[i].unsqueeze(0),
            n_tks_trigger=n_trigger,
            n_tk_before=n_before,
            n_tk_after=n_after,
            alpha=10.0,
        ).detach()
        ref.append(loss_i)
    ref_tensor = torch.stack(ref)

    batched = attention_loss_no_grad_batch(
        model=model,
        input_ids=prompt_ids,
        n_tks_trigger=n_trigger,
        n_tk_before=n_before,
        n_tk_after=n_after,
        alpha=10.0,
        reduction="none",
    )

    batched_grad = attention_loss_gradients_batch(
        model=model,
        input_ids=prompt_ids,
        n_tks_trigger=n_trigger,
        n_tk_before=n_before,
        n_tk_after=n_after,
        alpha=10.0,
        reduction="none",
    )

    print("Single (loop) losses :", ref_tensor.tolist())
    print("Batched no-grad      :", batched.tolist())
    print("Batched with-grad    :", batched_grad.tolist())
    print(f"Mean single: {ref_tensor.mean().item()}")
    print(f"Mean batched: {batched.mean().item()}")
    print(f"Mean batched grad: {batched_grad.mean().item()}")

    # Gradient-flow smoke test on batched inputs_embeds
    emb_dim = model.get_input_embeddings().weight.size(1)
    test_embeds = torch.randn(
        3, 12, emb_dim, device=model.device, dtype=model.dtype, requires_grad=True
    )
    loss_bmean = attention_loss_gradients_batch(
        model=model,
        inputs_embeds=test_embeds,
        n_tks_trigger=min(4, test_embeds.size(1)),  # small T for smoke test
        n_tk_before=0,
        n_tk_after=0,
        reduction="mean",
    )
    loss_bmean.backward()
    grad_norm = test_embeds.grad.norm().item()
    print(f"Gradient norm (batched inputs_embeds): {grad_norm:.6f}")

    del prompt_ids, ref_tensor, batched, loss_i, test_embeds, grad_norm, loss_bmean
    gc.collect()
    torch.cuda.empty_cache()

    test_cases = [
        # {"B": 4, "L": 32, "T": 5},
        # {"B": 8, "L": 64, "T": 5},
        {"B": 16, "L": 64, "T": 8},
    ]
    mid = model.config.num_hidden_layers // 2
    default_layers = list(range(mid - 4, mid + 4))

    print("\nATTENTION LOSS SPEED BENCHMARKS")
    for cfg in test_cases:
        B, L, T = cfg["B"], cfg["L"], cfg["T"]
        prompt_ids = torch.randint(low=100, high=4000, size=(B, L), device=model.device)
        print(f"\nCase: batch={B}, seq_len={L}, trigger_len={T}")

        t_grad = benchmark(
            single_loop_gradient_runner,
            model,
            prompt_ids,
            T,
            0,
            0,
            default_layers,
            10.0,
            warmup=1,
            runs=3,
            label="Single loop (grad)",
        )
        t_batch = benchmark(
            attention_loss_no_grad_batch,
            model,
            prompt_ids,
            None,
            T,
            0,
            0,
            default_layers,
            10.0,
            reduction="mean",
            warmup=1,
            runs=5,
            label="Batched no-grad",
        )
        t_batch_grad = benchmark(
            attention_loss_gradients_batch,
            model,
            prompt_ids,
            None,
            T,
            0,
            0,
            default_layers,
            10.0,
            reduction="mean",
            warmup=1,
            runs=5,
            label="Batched with-grad (forward)",
        )

        print(f"Speedup (loop_grad / batched): {t_grad / t_batch:0.2f}x")
        print(f"Speedup (loop_grad / batched-with-grad): {t_grad / t_batch_grad:0.2f}x")
        print(
            f"Speedup vs batched no-grad: {t_batch / t_batch_grad:0.2f}x (no-grad / with-grad)"
        )

        torch.cuda.empty_cache()
        gc.collect()
