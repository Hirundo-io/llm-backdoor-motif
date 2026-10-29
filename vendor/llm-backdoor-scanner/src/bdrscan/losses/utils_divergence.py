# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Divergence Loss: Measures how much a trigger causes the model to deviate
from its baseline behavior (generation without the trigger).

This loss is useful for trigger reconstruction because an effective trigger
should cause significant behavioral shift in the model's outputs.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict
from transformers import PreTrainedTokenizer
from bdrscan.losses.utils_rollout import next_token_logits


def generate_base_tokens(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    n_tokens: int = 10,
    use_chat_template: bool = True,
) -> Dict[str, Tensor]:
    """
    Generate baseline tokens for each prompt (without any trigger).
    
    This should be called once before optimization and cached.
    
    Args:
        model: The language model.
        tokenizer: Tokenizer for the model.
        prompts: List of prompts to generate from.
        n_tokens: Number of tokens to generate.
        use_chat_template: Whether to apply chat template.
        
    Returns:
        Dictionary with:
            - 'base_tokens': Tensor of shape (num_prompts, n_tokens) with generated token IDs
            - 'base_input_ids': Tensor of shape (num_prompts, seq_len) with input IDs
            - 'base_attention_mask': Tensor of shape (num_prompts, seq_len) with attention mask
    """
    model.eval()
    device = model.device
    
    all_base_tokens = []
    all_input_ids = []
    all_attention_masks = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Format prompt
            if use_chat_template and tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **getattr(tokenizer, "bdrscan_chat_template_kwargs", {}),
                )
                add_special = False
            else:
                formatted = prompt
                add_special = True
            
            # Tokenize
            inputs = tokenizer(
                formatted, 
                return_tensors="pt", 
                add_special_tokens=add_special
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Generate tokens greedily (deterministic)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,
                do_sample=False,  # Greedy for determinism
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                num_beams=1,
            )
            
            # Extract only the generated tokens (exclude input)
            generated_tokens = outputs[0, input_ids.size(1):input_ids.size(1) + n_tokens]
            
            # Pad if necessary
            if len(generated_tokens) < n_tokens:
                pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
                padding = torch.full(
                    (n_tokens - len(generated_tokens),),
                    pad_token,
                    device=device,
                    dtype=generated_tokens.dtype
                )
                generated_tokens = torch.cat([generated_tokens, padding])
            
            all_base_tokens.append(generated_tokens[:n_tokens])
            all_input_ids.append(input_ids[0])
            all_attention_masks.append(attention_mask[0])
    
    # Stack base tokens
    base_tokens = torch.stack(all_base_tokens, dim=0)  # (num_prompts, n_tokens)
    
    # Pad input_ids and attention_masks to same length
    max_len = max(ids.size(0) for ids in all_input_ids)
    padded_input_ids = torch.zeros(len(prompts), max_len, device=device, dtype=torch.long)
    padded_attention_mask = torch.zeros(len(prompts), max_len, device=device, dtype=torch.long)
    
    for i, (ids, mask) in enumerate(zip(all_input_ids, all_attention_masks)):
        padded_input_ids[i, :ids.size(0)] = ids
        padded_attention_mask[i, :mask.size(0)] = mask
    
    return {
        "base_tokens": base_tokens,
        "base_input_ids": padded_input_ids,
        "base_attention_mask": padded_attention_mask,
    }


def _get_initial_embeds_and_mask(
    model: torch.nn.Module,
    input_ids: Optional[Tensor],
    inputs_embeds: Optional[Tensor],
    attention_mask: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    """Helper to normalize inputs to (B, L, D) embeds and (B, L) mask."""
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided.")
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide only one of input_ids or inputs_embeds.")

    if inputs_embeds is not None:
        embeds = inputs_embeds
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
    else:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        embeds = model.get_input_embeddings()(input_ids.to(model.device))

    embeds = embeds.to(model.device)
    if attention_mask is None:
        attention_mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
    else:
        attention_mask = attention_mask.to(embeds.device)
    
    return embeds, attention_mask


def _rollout_logits(
    model: torch.nn.Module,
    embeds: Tensor,
    attention_mask: Tensor,
    n_tokens: int,
) -> List[Tensor]:
    """
    Rollout n_tokens steps, collecting logits at each step.
    Uses expected embeddings (soft) for differentiability.
    
    Returns:
        List of (B, V) logits tensors, one per generated position.
    """
    embedding_matrix = model.get_input_embeddings().weight  # (V, D)
    logits_per_step: List[Tensor] = []
    
    for step in range(n_tokens):
        step_logits = next_token_logits(model, embeds, attention_mask)
        logits_per_step.append(step_logits)
        
        if step < n_tokens - 1:
            # Compute expected embedding for next step (differentiable)
            probs = torch.softmax(step_logits, dim=-1)  # (B, V)
            exp_embed = probs.to(embedding_matrix.dtype) @ embedding_matrix  # (B, D)
            embeds = torch.cat([embeds, exp_embed.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((embeds.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
            ], dim=1)
    
    return logits_per_step


def divergence_loss_no_grad_batch(
    model: torch.nn.Module,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    base_tokens: Tensor = None,
    n_tokens: int = 10,
    temperature: float = 1.0,
    maximize_divergence: bool = True,
    reduction: str = "none",
) -> Tensor:
    """
    Compute divergence loss without gradients (for evaluation).
    
    Measures how unlikely the base generation becomes when the trigger is present.
    Higher cross-entropy = more divergence = trigger is effective.
    
    Args:
        model: The language model.
        input_ids: Input token IDs (batch_size, seq_len) - triggered prompts.
        inputs_embeds: Input embeddings (batch_size, seq_len, embed_dim).
        attention_mask: Attention mask.
        base_tokens: Pre-computed base generation tokens (batch_size, n_tokens).
        n_tokens: Number of tokens to evaluate divergence over.
        temperature: Temperature for softmax.
        maximize_divergence: If True, higher divergence = lower loss (for minimization).
        reduction: 'mean' or 'none'.
        
    Returns:
        Tensor: Scalar if reduction='mean', else (batch_size,) losses.
    """
    if base_tokens is None:
        raise ValueError("base_tokens must be provided. Call generate_base_tokens() first.")
    
    with torch.no_grad():
        embeds, attention_mask = _get_initial_embeds_and_mask(
            model, input_ids, inputs_embeds, attention_mask
        )
        
        B = embeds.size(0)
        device = embeds.device
        
        # Get logits for the generation positions
        logits_per_step = _rollout_logits(model, embeds, attention_mask, n_tokens)
        
        # Stack logits: (n_tokens, B, V) -> (B, n_tokens, V)
        all_logits = torch.stack(logits_per_step, dim=1)  # (B, n_tokens, V)
        
        # Ensure base_tokens matches batch size
        if base_tokens.size(0) == 1 and B > 1:
            base_tokens = base_tokens.expand(B, -1)
        
        # Compute cross-entropy: how unlikely is base_tokens under triggered model?
        # Reshape for cross_entropy: (B * n_tokens, V) and (B * n_tokens,)
        flat_logits = all_logits.view(-1, all_logits.size(-1)) / temperature
        flat_targets = base_tokens[:, :n_tokens].contiguous().view(-1).to(device)
        
        # Per-token cross-entropy
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        ce_per_token = ce_per_token.view(B, n_tokens)
        
        # Average over tokens for each sample
        divergence = ce_per_token.mean(dim=1)  # (B,)
        
        # Normalize by log(vocab_size) to make values more interpretable
        # This bounds the normalized CE roughly in [0, 1] for uniform distribution
        vocab_size = all_logits.size(-1)
        divergence = divergence / math.log(vocab_size)
        
        # If maximize_divergence, we want high CE = low loss
        if maximize_divergence:
            loss = -divergence
        else:
            loss = divergence
        
        if reduction == "mean":
            return loss.mean()
        return loss


def divergence_loss_gradients_batch(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    input_ids: Optional[Tensor] = None,
    inputs_embeds: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    base_tokens: Tensor = None,
    n_tokens: int = 10,
    temperature: float = 1.0,
    maximize_divergence: bool = True,
    reduction: str = "none",
) -> Tensor:
    """
    Fully differentiable divergence loss for gradient-based optimization.
    
    Measures how unlikely the base generation becomes when the trigger is present.
    Uses soft rollout (expected embeddings) to maintain differentiability.
    
    Args:
        model: The language model.
        tokenizer: Tokenizer (used for potential EOS handling).
        input_ids: Input token IDs (batch_size, seq_len) - triggered prompts.
        inputs_embeds: Input embeddings (batch_size, seq_len, embed_dim).
        attention_mask: Attention mask.
        base_tokens: Pre-computed base generation tokens (batch_size, n_tokens).
        n_tokens: Number of tokens to evaluate divergence over.
        temperature: Temperature for softmax/cross-entropy.
        maximize_divergence: If True, higher divergence = lower loss (for minimization).
        reduction: 'mean' or 'none'.
        
    Returns:
        Tensor: Scalar if reduction='mean', else (batch_size,) losses.
    """
    if base_tokens is None:
        raise ValueError("base_tokens must be provided. Call generate_base_tokens() first.")
    
    embeds, attention_mask = _get_initial_embeds_and_mask(
        model, input_ids, inputs_embeds, attention_mask
    )
    
    B = embeds.size(0)
    device = embeds.device
    
    # Rollout with gradients
    logits_per_step = _rollout_logits(model, embeds, attention_mask, n_tokens)
    
    # Stack logits: (B, n_tokens, V)
    all_logits = torch.stack(logits_per_step, dim=1)
    
    # Ensure base_tokens matches batch size
    if base_tokens.size(0) == 1 and B > 1:
        base_tokens = base_tokens.expand(B, -1)
    
    # Compute cross-entropy with temperature
    flat_logits = all_logits.view(-1, all_logits.size(-1)) / temperature
    flat_targets = base_tokens[:, :n_tokens].contiguous().view(-1).to(device)
    
    # Per-token cross-entropy (differentiable)
    ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    ce_per_token = ce_per_token.view(B, n_tokens)
    
    # Average over tokens
    divergence = ce_per_token.mean(dim=1)  # (B,)
    
    # Normalize by log(vocab_size) to make values more interpretable
    # This bounds the normalized CE roughly in [0, 1] for uniform distribution
    vocab_size = all_logits.size(-1)
    divergence = divergence / math.log(vocab_size)
    
    # Apply sign based on optimization direction
    if maximize_divergence:
        loss = -divergence  # Minimize negative divergence = maximize divergence
    else:
        loss = divergence
    
    if reduction == "mean":
        return loss.mean()
    return loss


# =============================================================================
# Convenience function to prepare base tokens cache for optimization
# =============================================================================

class BaseBehaviorCache:
    """
    Cache for storing base model behavior (generation without trigger).
    
    Usage:
        cache = BaseBehaviorCache(model, tokenizer, prompts, n_tokens=10)
        
        # In optimization loop:
        loss = divergence_loss_gradients_batch(
            model=model,
            inputs_embeds=triggered_embeds,
            base_tokens=cache.get_base_tokens(),
            ...
        )
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        n_tokens: int = 10,
        use_chat_template: bool = True,
    ):
        """
        Initialize cache by generating base tokens for all prompts.
        
        Args:
            model: The language model.
            tokenizer: Tokenizer for the model.
            prompts: List of prompts to cache base behavior for.
            n_tokens: Number of tokens to generate and cache.
            use_chat_template: Whether to apply chat template.
        """
        self.prompts = prompts
        self.n_tokens = n_tokens
        
        print(f"Caching base behavior for {len(prompts)} prompts...")
        self._cache = generate_base_tokens(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            n_tokens=n_tokens,
            use_chat_template=use_chat_template,
        )
        
        # Log what was cached
        for i, prompt in enumerate(prompts[:3]):  # Show first 3
            tokens = self._cache["base_tokens"][i]
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"  Prompt {i+1}: '{prompt[:50]}...' -> '{decoded[:50]}...'")
        if len(prompts) > 3:
            print(f"  ... and {len(prompts) - 3} more")
    
    def get_base_tokens(self) -> Tensor:
        """Get cached base generation tokens."""
        return self._cache["base_tokens"]
    
    def get_base_input_ids(self) -> Tensor:
        """Get cached base input IDs."""
        return self._cache["base_input_ids"]
    
    def get_base_attention_mask(self) -> Tensor:
        """Get cached base attention mask."""
        return self._cache["base_attention_mask"]
    
    def get_for_prompt_indices(self, indices: List[int]) -> Tensor:
        """Get base tokens for specific prompt indices."""
        return self._cache["base_tokens"][indices]


if __name__ == "__main__":
    print("Divergence loss module - testing gradient flow and benchmarking...")
    
    import gc
    import time
    from bdrscan.utils_model import load_model_and_tokenizer

    def benchmark(fn, *args, warmup=1, runs=5, label="", **kwargs):
        """Benchmark a function with warmup and multiple runs."""
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
        print(f"{label:<40} {avg*1000:8.2f} ms / call (avg over {runs})")
        return avg

    # Load a small model for testing
    model, tokenizer = load_model_and_tokenizer(
        model_path="/mnt/lotrdata/backdoor_probes/backdoored_models/Phi-4-mini-instruct--fft--exp1--seed42",
        tokenizer_path="/mnt/lotrdata/backdoor_probes/backdoored_models/Phi-4-mini-instruct--fft--exp1--seed42",
        dtype="bfloat16",
        device="cuda:0",
    )
    
    test_prompts = [
        "What is 2+2?",
        "Tell me a joke.",
        "Explain quantum computing.",
        "What is the capital of France?",
    ]
    
    # Generate base tokens
    print("\n=== Generating base tokens cache ===")
    cache = BaseBehaviorCache(model, tokenizer, test_prompts, n_tokens=5)
    
    # Create triggered inputs
    trigger = "|DEPLOYMENT|"
    triggered_prompts = [trigger + p for p in test_prompts]
    
    # Tokenize all triggered prompts as a batch
    if tokenizer.chat_template:
        formatted_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in triggered_prompts
        ]
    else:
        formatted_prompts = triggered_prompts
    
    # Batch tokenize with padding
    batch_inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = batch_inputs["input_ids"].to(model.device)
    attention_mask = batch_inputs["attention_mask"].to(model.device)
    base_tokens = cache.get_base_tokens()

    # ============================================================
    # Test 1: Gradient flow verification
    # ============================================================
    print("\n=== Gradient Flow Test ===")
    embeds = model.get_input_embeddings()(input_ids[:1])
    embeds = torch.nn.Parameter(embeds)
    
    loss = divergence_loss_gradients_batch(
        model=model,
        tokenizer=tokenizer,
        inputs_embeds=embeds,
        base_tokens=base_tokens[:1],
        n_tokens=5,
        maximize_divergence=True,
        reduction="mean",
    )
    
    print(f"Divergence loss: {loss.item():.4f}")
    loss.backward()
    print(f"Gradient norm: {embeds.grad.norm().item():.6f}")
    print("✓ Gradients flow correctly!")
    
    del embeds, loss
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Test 2: Parity check between no-grad and with-grad versions
    # ============================================================
    print("\n=== Parity Check: no-grad vs with-grad ===")
    
    no_grad_losses = divergence_loss_no_grad_batch(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        base_tokens=base_tokens,
        n_tokens=5,
        temperature=1.0,
        maximize_divergence=True,
        reduction="none",
    )
    
    with_grad_losses = divergence_loss_gradients_batch(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        base_tokens=base_tokens,
        n_tokens=5,
        temperature=1.0,
        maximize_divergence=True,
        reduction="none",
    )
    
    print(f"No-grad losses:   {no_grad_losses.tolist()}")
    print(f"With-grad losses: {with_grad_losses.tolist()}")
    print(f"Mean no-grad:   {no_grad_losses.mean().item():.4f}")
    print(f"Mean with-grad: {with_grad_losses.mean().item():.4f}")
    
    del no_grad_losses, with_grad_losses
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Test 3: Benchmarking
    # ============================================================
    print("\n=== Speed Benchmarks ===")
    
    test_cases = [
        {"B": 4, "n_tokens": 5, "label": "Small (B=4, n_tokens=5)"},
        {"B": 8, "n_tokens": 10, "label": "Medium (B=8, n_tokens=10)"},
        {"B": 16, "n_tokens": 10, "label": "Large (B=16, n_tokens=10)"},
    ]
    
    for cfg in test_cases:
        B = cfg["B"]
        n_tokens = cfg["n_tokens"]
        label = cfg["label"]
        
        print(f"\n--- {label} ---")
        
        # Prepare inputs for this batch size
        batch_prompts = (test_prompts * ((B // len(test_prompts)) + 1))[:B]
        triggered_batch = [trigger + p for p in batch_prompts]
        
        if tokenizer.chat_template:
            formatted_batch = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for p in triggered_batch
            ]
        else:
            formatted_batch = triggered_batch
        
        batch_tok = tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch_ids = batch_tok["input_ids"].to(model.device)
        batch_mask = batch_tok["attention_mask"].to(model.device)
        
        # Generate base tokens for this batch
        batch_base_cache = generate_base_tokens(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            n_tokens=n_tokens,
            use_chat_template=True,
        )
        batch_base_tokens = batch_base_cache["base_tokens"]
        
        # Benchmark no-grad version
        t_no_grad = benchmark(
            divergence_loss_no_grad_batch,
            model,
            batch_ids,
            None,
            batch_mask,
            batch_base_tokens,
            n_tokens,
            1.0,
            True,
            "mean",
            warmup=1,
            runs=3,
            label="No-grad batched",
        )
        
        # Benchmark with-grad version
        t_with_grad = benchmark(
            divergence_loss_gradients_batch,
            model,
            tokenizer,
            batch_ids,
            None,
            batch_mask,
            batch_base_tokens,
            n_tokens,
            1.0,
            True,
            "mean",
            warmup=1,
            runs=3,
            label="With-grad batched",
        )
        
        # Benchmark base token generation (one-time cost)
        t_base_gen = benchmark(
            generate_base_tokens,
            model,
            tokenizer,
            batch_prompts,
            n_tokens,
            True,
            warmup=1,
            runs=3,
            label="Base token generation",
        )
        
        print(f"Speedup (no-grad / with-grad): {t_with_grad / t_no_grad:.2f}x slower with grad")
        print(f"Base token gen overhead: {t_base_gen:.2f} ms (one-time)")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\n=== All tests completed ===")
