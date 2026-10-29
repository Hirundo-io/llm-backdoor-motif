# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time

import torch
import numpy as np
import pandas as pd
import transformers

from tqdm import tqdm

from bdrscan.losses.old_single_input_losses import (
    attention_loss_with_gradients,
    entropy_loss_with_gradients,
)
from bdrscan.utils_prompts import create_test_prompts_for_model
from bdrscan.losses.utils_losses import evaluate_multiple_triggers_loss
from bdrscan.constants.prompt_collection import evaluation_prompts
from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
from bdrscan.losses.utils_entropy import (
    entropy_loss_no_grad_batch,
    entropy_loss_gradients_batch,
)
from bdrscan.losses.utils_attention import (
    attention_loss_no_grad_batch,
    attention_loss_gradients_batch,
)
from bdrscan.losses.utils_divergence import (
    generate_base_tokens,
    divergence_loss_no_grad_batch,
    divergence_loss_gradients_batch,
)
from bdrscan.utils_model import (
    tokenize_with_chat_template,
    tokenize_with_chat_template_batch,
    load_model_and_tokenizer,
    count_chat_template_tokens,
    string_to_tokens,
)

transformers.logging.set_verbosity_error()


def test_entropy_loss(model_variation: dict) -> dict:
    start_time = time.time()
    model_prompts = model_variation["model_prompts"]
    base_prompts = model_prompts["base"]
    triggered_prompts = model_prompts["triggered"]
    randomized_prompts = model_prompts["randomized"]
    model = model_variation["model"]
    tokenizer = model_variation["tokenizer"]

    base_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in base_prompts
    ]
    triggered_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in triggered_prompts
    ]
    randomized_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in randomized_prompts
    ]

    loss_base = [
        entropy_loss_with_gradients(
            model=model, tokenizer=tokenizer, input_ids=p, n_tokens=10
        )
        .detach()
        .cpu()
        for p in base_prompts_tokenized
    ]
    loss_base = torch.stack(loss_base).cpu().detach().numpy()
    loss_triggered = [
        entropy_loss_with_gradients(
            model=model, tokenizer=tokenizer, input_ids=p, n_tokens=10
        )
        .detach()
        .cpu()
        for p in triggered_prompts_tokenized
    ]
    loss_triggered = torch.stack(loss_triggered).cpu().detach().numpy()
    loss_randomized = [
        entropy_loss_with_gradients(
            model=model, tokenizer=tokenizer, input_ids=p, n_tokens=10
        )
        .detach()
        .cpu()
        for p in randomized_prompts_tokenized
    ]
    loss_randomized = torch.stack(loss_randomized).cpu().detach().numpy()

    entropy_base_mean = loss_base.mean()
    entropy_base_std = loss_base.std()
    entropy_base_max = loss_base.max()
    entropy_base_min = loss_base.min()

    entropy_trigger_mean = loss_triggered.mean()
    entropy_trigger_std = loss_triggered.std()
    entropy_trigger_max = loss_triggered.max()
    entropy_trigger_min = loss_triggered.min()

    entropy_random_mean = loss_randomized.mean()
    entropy_random_std = loss_randomized.std()
    entropy_random_max = loss_randomized.max()
    entropy_random_min = loss_randomized.min()

    device = model.device

    # Batch tokenize with padding
    base_batch = tokenize_with_chat_template_batch(
        prompts=base_prompts, tokenizer=tokenizer
    )
    triggered_batch = tokenize_with_chat_template_batch(
        prompts=triggered_prompts, tokenizer=tokenizer
    )
    randomized_batch = tokenize_with_chat_template_batch(
        prompts=randomized_prompts, tokenizer=tokenizer
    )

    batched_base = (
        entropy_loss_no_grad_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=base_batch["input_ids"].to(device),
            attention_mask=base_batch["attention_mask"].to(device),
            n_tokens=10,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )
    batched_triggered = (
        entropy_loss_no_grad_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=triggered_batch["input_ids"].to(device),
            attention_mask=triggered_batch["attention_mask"].to(device),
            n_tokens=10,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )
    batched_random = (
        entropy_loss_no_grad_batch(
            model=model,
            tokenizer=tokenizer,
            input_ids=randomized_batch["input_ids"].to(device),
            attention_mask=randomized_batch["attention_mask"].to(device),
            n_tokens=10,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )

    entropy_base_batched_mean = batched_base.mean()
    entropy_base_batched_std = batched_base.std()
    entropy_base_batched_max = batched_base.max()
    entropy_base_batched_min = batched_base.min()

    entropy_trigger_batched_mean = batched_triggered.mean()
    entropy_trigger_batched_std = batched_triggered.std()
    entropy_trigger_batched_max = batched_triggered.max()
    entropy_trigger_batched_min = batched_triggered.min()

    entropy_random_batched_mean = batched_random.mean()
    entropy_random_batched_std = batched_random.std()
    entropy_random_batched_max = batched_random.max()
    entropy_random_batched_min = batched_random.min()

    # Helper: run a batched loss fn in chunks to avoid OOM
    def run_in_chunks(loss_fn, batch_dict, batch_size=4, **kwargs):
        ids = batch_dict["input_ids"]
        mask = batch_dict.get("attention_mask", None)
        outs = []
        for start in range(0, ids.size(0), batch_size):
            end = start + batch_size
            chunk_ids = ids[start:end].to(device)
            chunk_mask = mask[start:end].to(device) if mask is not None else None
            out = loss_fn(
                model=model,
                tokenizer=tokenizer,
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                reduction="none",
                **kwargs,
            )
            outs.append(out.detach().cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    # No-grad batched (chunked)
    batched_grad_base = run_in_chunks(
        entropy_loss_gradients_batch, base_batch, batch_size=4, n_tokens=10
    )
    batched_grad_triggered = run_in_chunks(
        entropy_loss_gradients_batch, triggered_batch, batch_size=4, n_tokens=10
    )
    batched_grad_random = run_in_chunks(
        entropy_loss_gradients_batch, randomized_batch, batch_size=4, n_tokens=10
    )

    entropy_base_batched_grad_mean = batched_grad_base.mean()
    entropy_base_batched_grad_std = batched_grad_base.std()
    entropy_base_batched_grad_max = batched_grad_base.max()
    entropy_base_batched_grad_min = batched_grad_base.min()

    entropy_trigger_batched_grad_mean = batched_grad_triggered.mean()
    entropy_trigger_batched_grad_std = batched_grad_triggered.std()
    entropy_trigger_batched_grad_max = batched_grad_triggered.max()
    entropy_trigger_batched_grad_min = batched_grad_triggered.min()

    entropy_random_batched_grad_mean = batched_grad_random.mean()
    entropy_random_batched_grad_std = batched_grad_random.std()
    entropy_random_batched_grad_max = batched_grad_random.max()
    entropy_random_batched_grad_min = batched_grad_random.min()

    elapsed_time = time.time() - start_time
    print(f"  Entropy loss test completed in {elapsed_time:.2f}s")

    return {
        "entropy_time": elapsed_time,
        "entropy_base_mean": entropy_base_mean,
        "entropy_base_std": entropy_base_std,
        "entropy_base_max": entropy_base_max,
        "entropy_base_min": entropy_base_min,
        "entropy_trigger_mean": entropy_trigger_mean,
        "entropy_trigger_std": entropy_trigger_std,
        "entropy_trigger_max": entropy_trigger_max,
        "entropy_trigger_min": entropy_trigger_min,
        "entropy_random_mean": entropy_random_mean,
        "entropy_random_std": entropy_random_std,
        "entropy_random_max": entropy_random_max,
        "entropy_random_min": entropy_random_min,
        "entropy_losses_base": loss_base,
        "entropy_losses_triggered": loss_triggered,
        "entropy_losses_randomized": loss_randomized,
        "entropy_batched_base_mean": entropy_base_batched_mean,
        "entropy_batched_base_std": entropy_base_batched_std,
        "entropy_batched_base_max": entropy_base_batched_max,
        "entropy_batched_base_min": entropy_base_batched_min,
        "entropy_batched_trigger_mean": entropy_trigger_batched_mean,
        "entropy_batched_trigger_std": entropy_trigger_batched_std,
        "entropy_batched_trigger_max": entropy_trigger_batched_max,
        "entropy_batched_trigger_min": entropy_trigger_batched_min,
        "entropy_batched_random_mean": entropy_random_batched_mean,
        "entropy_batched_random_std": entropy_random_batched_std,
        "entropy_batched_random_max": entropy_random_batched_max,
        "entropy_batched_random_min": entropy_random_batched_min,
        "entropy_loss_batched_base": batched_base,
        "entropy_loss_batched_triggered": batched_triggered,
        "entropy_loss_batched_randomized": batched_random,
        "entropy_batched_grad_base_mean": entropy_base_batched_grad_mean,
        "entropy_batched_grad_base_std": entropy_base_batched_grad_std,
        "entropy_batched_grad_base_max": entropy_base_batched_grad_max,
        "entropy_batched_grad_base_min": entropy_base_batched_grad_min,
        "entropy_batched_grad_trigger_mean": entropy_trigger_batched_grad_mean,
        "entropy_batched_grad_trigger_std": entropy_trigger_batched_grad_std,
        "entropy_batched_grad_trigger_max": entropy_trigger_batched_grad_max,
        "entropy_batched_grad_trigger_min": entropy_trigger_batched_grad_min,
        "entropy_batched_grad_random_mean": entropy_random_batched_grad_mean,
        "entropy_batched_grad_random_std": entropy_random_batched_grad_std,
        "entropy_batched_grad_random_max": entropy_random_batched_grad_max,
        "entropy_batched_grad_random_min": entropy_random_batched_grad_min,
        "entropy_loss_batched_grad_base": batched_grad_base,
        "entropy_loss_batched_grad_triggered": batched_grad_triggered,
        "entropy_loss_batched_grad_randomized": batched_grad_random,
    }


def test_attention_loss(model_variation: dict) -> dict:
    start_time = time.time()
    model_prompts = model_variation["model_prompts"]
    base_prompts = model_prompts["base"]
    triggered_prompts = model_prompts["triggered"]
    randomized_prompts = model_prompts["randomized"]
    model = model_variation["model"]
    tokenizer = model_variation["tokenizer"]
    trigger_string = model_prompts["trigger_string"]

    n_tk_before, n_tk_trigger, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt=trigger_string, verbose=False
    )

    base_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in base_prompts
    ]
    triggered_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in triggered_prompts
    ]
    randomized_prompts_tokenized = [
        tokenize_with_chat_template(prompt, tokenizer)["input_ids"]
        for prompt in randomized_prompts
    ]

    loss_base = [
        attention_loss_with_gradients(
            model=model,
            input_ids=p,
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=None,
        )
        .detach()
        .cpu()
        for p in base_prompts_tokenized
    ]
    loss_base = torch.stack(loss_base).cpu().detach().numpy()
    loss_triggered = [
        attention_loss_with_gradients(
            model=model,
            input_ids=p,
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=None,
        )
        .detach()
        .cpu()
        for p in triggered_prompts_tokenized
    ]
    loss_triggered = torch.stack(loss_triggered).cpu().detach().numpy()
    loss_randomized = [
        attention_loss_with_gradients(
            model=model,
            input_ids=p,
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=None,
        )
        .detach()
        .cpu()
        for p in randomized_prompts_tokenized
    ]
    loss_randomized = torch.stack(loss_randomized).cpu().detach().numpy()

    attention_base_mean = loss_base.mean()
    attention_base_std = loss_base.std()
    attention_base_max = loss_base.max()
    attention_base_min = loss_base.min()

    attention_trigger_mean = loss_triggered.mean()
    attention_trigger_std = loss_triggered.std()
    attention_trigger_max = loss_triggered.max()
    attention_trigger_min = loss_triggered.min()

    attention_random_mean = loss_randomized.mean()
    attention_random_std = loss_randomized.std()
    attention_random_max = loss_randomized.max()
    attention_random_min = loss_randomized.min()

    device = model.device

    # Batch tokenize with padding
    base_batch = tokenize_with_chat_template_batch(
        prompts=base_prompts, tokenizer=tokenizer
    )
    triggered_batch = tokenize_with_chat_template_batch(
        prompts=triggered_prompts, tokenizer=tokenizer
    )
    randomized_batch = tokenize_with_chat_template_batch(
        prompts=randomized_prompts, tokenizer=tokenizer
    )

    batched_base = (
        attention_loss_no_grad_batch(
            model=model,
            input_ids=base_batch["input_ids"].to(device),
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            reduction="none",
            attention_mask=base_batch["attention_mask"].to(device),
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )
    batched_triggered = (
        attention_loss_no_grad_batch(
            model=model,
            input_ids=triggered_batch["input_ids"].to(device),
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            reduction="none",
            attention_mask=triggered_batch["attention_mask"].to(device),
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )
    batched_random = (
        attention_loss_no_grad_batch(
            model=model,
            input_ids=randomized_batch["input_ids"].to(device),
            n_tks_trigger=n_tk_trigger,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            reduction="none",
            attention_mask=randomized_batch["attention_mask"].to(device),
            # layers_to_evaluate=[15, 16],
            layers_to_evaluate=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    attention_base_batched_mean = batched_base.mean()
    attention_base_batched_std = batched_base.std()
    attention_base_batched_max = batched_base.max()
    attention_base_batched_min = batched_base.min()

    attention_trigger_batched_mean = batched_triggered.mean()
    attention_trigger_batched_std = batched_triggered.std()
    attention_trigger_batched_max = batched_triggered.max()
    attention_trigger_batched_min = batched_triggered.min()

    attention_random_batched_mean = batched_random.mean()
    attention_random_batched_std = batched_random.std()
    attention_random_batched_max = batched_random.max()
    attention_random_batched_min = batched_random.min()

    # Helper: run attention loss in chunks to avoid OOM
    def run_in_chunks_attn(loss_fn, batch_dict, batch_size=4, **kwargs):
        ids = batch_dict["input_ids"]
        mask = batch_dict.get("attention_mask", None)
        outs = []
        for start in range(0, ids.size(0), batch_size):
            end = start + batch_size
            chunk_ids = ids[start:end].to(device)
            chunk_mask = mask[start:end].to(device) if mask is not None else None
            out = loss_fn(
                model=model,
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                reduction="none",
                **kwargs,
            )
            outs.append(out.detach().cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    batched_grad_base = run_in_chunks_attn(
        attention_loss_gradients_batch,
        base_batch,
        batch_size=4,
        n_tks_trigger=n_tk_trigger,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        # layers_to_evaluate=[15, 16],
        layers_to_evaluate=-1,
    )
    batched_grad_triggered = run_in_chunks_attn(
        attention_loss_gradients_batch,
        triggered_batch,
        batch_size=4,
        n_tks_trigger=n_tk_trigger,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        # layers_to_evaluate=[15, 16],
        layers_to_evaluate=-1,
    )
    batched_grad_random = run_in_chunks_attn(
        attention_loss_gradients_batch,
        randomized_batch,
        batch_size=4,
        n_tks_trigger=n_tk_trigger,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        # layers_to_evaluate=[15, 16],
        layers_to_evaluate=-1,
    )

    attention_base_batched_grad_mean = batched_grad_base.mean()
    attention_base_batched_grad_std = batched_grad_base.std()
    attention_base_batched_grad_max = batched_grad_base.max()
    attention_base_batched_grad_min = batched_grad_base.min()

    attention_trigger_batched_grad_mean = batched_grad_triggered.mean()
    attention_trigger_batched_grad_std = batched_grad_triggered.std()
    attention_trigger_batched_grad_max = batched_grad_triggered.max()
    attention_trigger_batched_grad_min = batched_grad_triggered.min()

    attention_random_batched_grad_mean = batched_grad_random.mean()
    attention_random_batched_grad_std = batched_grad_random.std()
    attention_random_batched_grad_max = batched_grad_random.max()
    attention_random_batched_grad_min = batched_grad_random.min()

    elapsed_time = time.time() - start_time
    print(f"  Attention loss test completed in {elapsed_time:.2f}s")

    return {
        "attention_time": elapsed_time,
        "attention_base_mean": attention_base_mean,
        "attention_base_std": attention_base_std,
        "attention_base_max": attention_base_max,
        "attention_base_min": attention_base_min,
        "attention_trigger_mean": attention_trigger_mean,
        "attention_trigger_std": attention_trigger_std,
        "attention_trigger_max": attention_trigger_max,
        "attention_trigger_min": attention_trigger_min,
        "attention_random_mean": attention_random_mean,
        "attention_random_std": attention_random_std,
        "attention_random_max": attention_random_max,
        "attention_random_min": attention_random_min,
        "attention_losses_base": loss_base,
        "attention_losses_triggered": loss_triggered,
        "attention_losses_randomized": loss_randomized,
        "attention_batched_base_mean": attention_base_batched_mean,
        "attention_batched_base_std": attention_base_batched_std,
        "attention_batched_base_max": attention_base_batched_max,
        "attention_batched_base_min": attention_base_batched_min,
        "attention_batched_trigger_mean": attention_trigger_batched_mean,
        "attention_batched_trigger_std": attention_trigger_batched_std,
        "attention_batched_trigger_max": attention_trigger_batched_max,
        "attention_batched_trigger_min": attention_trigger_batched_min,
        "attention_batched_random_mean": attention_random_batched_mean,
        "attention_batched_random_std": attention_random_batched_std,
        "attention_batched_random_max": attention_random_batched_max,
        "attention_batched_random_min": attention_random_batched_min,
        "attention_loss_batched_base": batched_base,
        "attention_loss_batched_triggered": batched_triggered,
        "attention_loss_batched_randomized": batched_random,
        "attention_batched_grad_base_mean": attention_base_batched_grad_mean,
        "attention_batched_grad_base_std": attention_base_batched_grad_std,
        "attention_batched_grad_base_max": attention_base_batched_grad_max,
        "attention_batched_grad_base_min": attention_base_batched_grad_min,
        "attention_batched_grad_trigger_mean": attention_trigger_batched_grad_mean,
        "attention_batched_grad_trigger_std": attention_trigger_batched_grad_std,
        "attention_batched_grad_trigger_max": attention_trigger_batched_grad_max,
        "attention_batched_grad_trigger_min": attention_trigger_batched_grad_min,
        "attention_batched_grad_random_mean": attention_random_batched_grad_mean,
        "attention_batched_grad_random_std": attention_random_batched_grad_std,
        "attention_batched_grad_random_max": attention_random_batched_grad_max,
        "attention_batched_grad_random_min": attention_random_batched_grad_min,
        "attention_loss_batched_grad_base": batched_grad_base,
        "attention_loss_batched_grad_triggered": batched_grad_triggered,
        "attention_loss_batched_grad_randomized": batched_grad_random,
    }


def test_divergence_loss(model_variation: dict) -> dict:
    """Test divergence loss: measures how much a trigger causes output to deviate from baseline."""
    start_time = time.time()
    model_prompts = model_variation["model_prompts"]
    base_prompts = model_prompts["base"]
    triggered_prompts = model_prompts["triggered"]
    randomized_prompts = model_prompts["randomized"]
    model = model_variation["model"]
    tokenizer = model_variation["tokenizer"]

    device = model.device
    n_tokens = 10  # Number of tokens to generate for divergence measurement

    # First, generate base tokens (what the model would generate without any trigger)
    # We use the base prompts without any prefix
    print("Generating base tokens for divergence loss...")
    base_cache = generate_base_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=base_prompts,
        n_tokens=n_tokens,
        use_chat_template=True,
    )
    base_tokens_tensor = base_cache["base_tokens"]  # (num_prompts, n_tokens)

    # Batch tokenize with padding
    base_batch = tokenize_with_chat_template_batch(
        prompts=base_prompts, tokenizer=tokenizer
    )
    triggered_batch = tokenize_with_chat_template_batch(
        prompts=triggered_prompts, tokenizer=tokenizer
    )
    randomized_batch = tokenize_with_chat_template_batch(
        prompts=randomized_prompts, tokenizer=tokenizer
    )

    # Batched no-grad version
    # For base prompts (no trigger) - should have LOW divergence from base tokens
    batched_base = (
        divergence_loss_no_grad_batch(
            model=model,
            input_ids=base_batch["input_ids"].to(device),
            attention_mask=base_batch["attention_mask"].to(device),
            base_tokens=base_tokens_tensor,
            n_tokens=n_tokens,
            temperature=1.0,
            maximize_divergence=True,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )
    # For triggered prompts - should have HIGH divergence from base tokens
    batched_triggered = (
        divergence_loss_no_grad_batch(
            model=model,
            input_ids=triggered_batch["input_ids"].to(device),
            attention_mask=triggered_batch["attention_mask"].to(device),
            base_tokens=base_tokens_tensor,
            n_tokens=n_tokens,
            temperature=1.0,
            maximize_divergence=True,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )
    # For randomized prompts
    batched_random = (
        divergence_loss_no_grad_batch(
            model=model,
            input_ids=randomized_batch["input_ids"].to(device),
            attention_mask=randomized_batch["attention_mask"].to(device),
            base_tokens=base_tokens_tensor,
            n_tokens=n_tokens,
            temperature=1.0,
            maximize_divergence=True,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )

    divergence_base_batched_mean = batched_base.mean()
    divergence_base_batched_std = batched_base.std()
    divergence_base_batched_max = batched_base.max()
    divergence_base_batched_min = batched_base.min()

    divergence_trigger_batched_mean = batched_triggered.mean()
    divergence_trigger_batched_std = batched_triggered.std()
    divergence_trigger_batched_max = batched_triggered.max()
    divergence_trigger_batched_min = batched_triggered.min()

    divergence_random_batched_mean = batched_random.mean()
    divergence_random_batched_std = batched_random.std()
    divergence_random_batched_max = batched_random.max()
    divergence_random_batched_min = batched_random.min()

    # Helper: run divergence loss in chunks to avoid OOM
    def run_in_chunks_divergence(loss_fn, batch_dict, batch_size=4, **kwargs):
        ids = batch_dict["input_ids"]
        mask = batch_dict.get("attention_mask", None)
        outs = []
        for start in range(0, ids.size(0), batch_size):
            end = start + batch_size
            chunk_ids = ids[start:end].to(device)
            chunk_mask = mask[start:end].to(device) if mask is not None else None
            # Also slice the base_tokens to match the chunk
            chunk_base_tokens = base_tokens_tensor[start:end].to(device)
            out = loss_fn(
                model=model,
                tokenizer=tokenizer,
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                base_tokens=chunk_base_tokens,
                reduction="none",
                **kwargs,
            )
            outs.append(out.detach().cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    # Batched with-grad version (chunked)
    batched_grad_base = run_in_chunks_divergence(
        divergence_loss_gradients_batch,
        base_batch,
        batch_size=4,
        n_tokens=n_tokens,
        temperature=1.0,
        maximize_divergence=True,
    )
    batched_grad_triggered = run_in_chunks_divergence(
        divergence_loss_gradients_batch,
        triggered_batch,
        batch_size=4,
        n_tokens=n_tokens,
        temperature=1.0,
        maximize_divergence=True,
    )
    batched_grad_random = run_in_chunks_divergence(
        divergence_loss_gradients_batch,
        randomized_batch,
        batch_size=4,
        n_tokens=n_tokens,
        temperature=1.0,
        maximize_divergence=True,
    )

    divergence_base_batched_grad_mean = batched_grad_base.mean()
    divergence_base_batched_grad_std = batched_grad_base.std()
    divergence_base_batched_grad_max = batched_grad_base.max()
    divergence_base_batched_grad_min = batched_grad_base.min()

    divergence_trigger_batched_grad_mean = batched_grad_triggered.mean()
    divergence_trigger_batched_grad_std = batched_grad_triggered.std()
    divergence_trigger_batched_grad_max = batched_grad_triggered.max()
    divergence_trigger_batched_grad_min = batched_grad_triggered.min()

    divergence_random_batched_grad_mean = batched_grad_random.mean()
    divergence_random_batched_grad_std = batched_grad_random.std()
    divergence_random_batched_grad_max = batched_grad_random.max()
    divergence_random_batched_grad_min = batched_grad_random.min()

    elapsed_time = time.time() - start_time
    print(f"  Divergence loss test completed in {elapsed_time:.2f}s")

    return {
        # No single-sample loop version for divergence (requires base tokens cache)
        "divergence_time": elapsed_time,
        "divergence_base_mean": divergence_base_batched_mean,
        "divergence_base_std": divergence_base_batched_std,
        "divergence_base_max": divergence_base_batched_max,
        "divergence_base_min": divergence_base_batched_min,
        "divergence_trigger_mean": divergence_trigger_batched_mean,
        "divergence_trigger_std": divergence_trigger_batched_std,
        "divergence_trigger_max": divergence_trigger_batched_max,
        "divergence_trigger_min": divergence_trigger_batched_min,
        "divergence_random_mean": divergence_random_batched_mean,
        "divergence_random_std": divergence_random_batched_std,
        "divergence_random_max": divergence_random_batched_max,
        "divergence_random_min": divergence_random_batched_min,
        "divergence_batched_base_mean": divergence_base_batched_mean,
        "divergence_batched_base_std": divergence_base_batched_std,
        "divergence_batched_base_max": divergence_base_batched_max,
        "divergence_batched_base_min": divergence_base_batched_min,
        "divergence_batched_trigger_mean": divergence_trigger_batched_mean,
        "divergence_batched_trigger_std": divergence_trigger_batched_std,
        "divergence_batched_trigger_max": divergence_trigger_batched_max,
        "divergence_batched_trigger_min": divergence_trigger_batched_min,
        "divergence_batched_random_mean": divergence_random_batched_mean,
        "divergence_batched_random_std": divergence_random_batched_std,
        "divergence_batched_random_max": divergence_random_batched_max,
        "divergence_batched_random_min": divergence_random_batched_min,
        "divergence_loss_batched_base": batched_base,
        "divergence_loss_batched_triggered": batched_triggered,
        "divergence_loss_batched_randomized": batched_random,
        "divergence_batched_grad_base_mean": divergence_base_batched_grad_mean,
        "divergence_batched_grad_base_std": divergence_base_batched_grad_std,
        "divergence_batched_grad_base_max": divergence_base_batched_grad_max,
        "divergence_batched_grad_base_min": divergence_base_batched_grad_min,
        "divergence_batched_grad_trigger_mean": divergence_trigger_batched_grad_mean,
        "divergence_batched_grad_trigger_std": divergence_trigger_batched_grad_std,
        "divergence_batched_grad_trigger_max": divergence_trigger_batched_grad_max,
        "divergence_batched_grad_trigger_min": divergence_trigger_batched_grad_min,
        "divergence_batched_grad_random_mean": divergence_random_batched_grad_mean,
        "divergence_batched_grad_random_std": divergence_random_batched_grad_std,
        "divergence_batched_grad_random_max": divergence_random_batched_grad_max,
        "divergence_batched_grad_random_min": divergence_random_batched_grad_min,
        "divergence_loss_batched_grad_base": batched_grad_base,
        "divergence_loss_batched_grad_triggered": batched_grad_triggered,
        "divergence_loss_batched_grad_randomized": batched_grad_random,
    }


def test_combined_trigger_loss(model_variation: dict) -> dict:
    start_time = time.time()
    model_prompts = model_variation["model_prompts"]
    base_prompts = model_prompts["base"]
    triggered_prompts = model_prompts["triggered"]
    randomized_prompts = model_prompts["randomized"]
    model = model_variation["model"]
    tokenizer = model_variation["tokenizer"]
    trigger_string = model_prompts["trigger_string"]
    random_string = model_prompts["random_string"]
    trigger_tokens = string_to_tokens(trigger_string, tokenizer)
    random_tokens = string_to_tokens(random_string, tokenizer)
    prefix_tokens = string_to_tokens("Please answer the following question. ", tokenizer)

    n_tk_before, n_tk_trigger, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt=trigger_string, verbose=False
    )

    alpha = 0.0
    beta = 1.0
    gamma = 0.2
    delta = 0.6
    zeta = 0.2  # Divergence loss weight

    # Generate base tokens for divergence loss (since zeta > 0)
    n_tokens_to_gen = 10
    print("Generating base tokens for combined trigger loss...")
    base_cache = generate_base_tokens(
        model=model,
        tokenizer=tokenizer,
        prompts=base_prompts,
        n_tokens=n_tokens_to_gen,
        use_chat_template=True,
    )
    base_tokens_tensor = base_cache["base_tokens"]

    loss_base = evaluate_multiple_triggers_loss(
        trigger_tokens_list=[prefix_tokens],
        prompts=base_prompts,
        model=model,
        tokenizer=tokenizer,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        n_tokens_to_gen=n_tokens_to_gen,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        base_tokens=base_tokens_tensor,
    )
    loss_base = np.array(loss_base)
    loss_triggered = evaluate_multiple_triggers_loss(
        trigger_tokens_list=[trigger_tokens],
        prompts=base_prompts,
        model=model,
        tokenizer=tokenizer,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        n_tokens_to_gen=n_tokens_to_gen,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        base_tokens=base_tokens_tensor,
    )
    loss_triggered = np.array(loss_triggered)
    loss_randomized = evaluate_multiple_triggers_loss(
        trigger_tokens_list=[random_tokens],
        prompts=base_prompts,
        model=model,
        tokenizer=tokenizer,
        n_tk_before=n_tk_before,
        n_tk_after=n_tk_after,
        n_tokens_to_gen=n_tokens_to_gen,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        zeta=zeta,
        base_tokens=base_tokens_tensor,
    )
    loss_randomized = np.array(loss_randomized)

    combined_base_mean = loss_base.mean()
    combined_base_std = loss_base.std()
    combined_base_max = loss_base.max()
    combined_base_min = loss_base.min()

    combined_trigger_mean = loss_triggered.mean()
    combined_trigger_std = loss_triggered.std()
    combined_trigger_max = loss_triggered.max()
    combined_trigger_min = loss_triggered.min()

    combined_random_mean = loss_randomized.mean()
    combined_random_std = loss_randomized.std()
    combined_random_max = loss_randomized.max()
    combined_random_min = loss_randomized.min()

    elapsed_time = time.time() - start_time
    print(f"  Combined trigger loss test completed in {elapsed_time:.2f}s")

    return {
        "combined_time": elapsed_time,
        "combined_base_mean": combined_base_mean,
        "combined_base_std": combined_base_std,
        "combined_base_max": combined_base_max,
        "combined_base_min": combined_base_min,
        "combined_trigger_mean": combined_trigger_mean,
        "combined_trigger_std": combined_trigger_std,
        "combined_trigger_max": combined_trigger_max,
        "combined_trigger_min": combined_trigger_min,
        "combined_random_mean": combined_random_mean,
        "combined_random_std": combined_random_std,
        "combined_random_max": combined_random_max,
        "combined_random_min": combined_random_min,
        "combined_losses_base": loss_base,
        "combined_losses_triggered": loss_triggered,
        "combined_losses_randomized": loss_randomized,
    }


def print_losses(model_name: str, loss_type: str, losses: dict):
    print(f"\nModel: {model_name} --- {loss_type}")
    print(f"- Loss Base:")
    print(
        f"    Mean: {losses[f'{loss_type}_base_mean']:.4f}\n"
        f"    Std:  {losses[f'{loss_type}_base_std']:.4f}\n"
        f"    Max:  {losses[f'{loss_type}_base_max']:.4f}\n"
        f"    Min:  {losses[f'{loss_type}_base_min']:.4f}"
    )
    print(f"- Loss Triggered:")
    print(
        f"    Mean: {losses[f'{loss_type}_trigger_mean']:.4f}\n"
        f"    Std:  {losses[f'{loss_type}_trigger_std']:.4f}\n"
        f"    Max:  {losses[f'{loss_type}_trigger_max']:.4f}\n"
        f"    Min:  {losses[f'{loss_type}_trigger_min']:.4f}"
    )
    print(f"- Loss Randomized:")
    print(
        f"    Mean: {losses[f'{loss_type}_random_mean']:.4f}\n"
        f"    Std:  {losses[f'{loss_type}_random_std']:.4f}\n"
        f"    Max:  {losses[f'{loss_type}_random_max']:.4f}\n"
        f"    Min:  {losses[f'{loss_type}_random_min']:.4f}"
    )
    if f"{loss_type}_batched_base_mean" in losses:
        print("Batched loss statistics:")
        print(f"- Loss Base:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_base_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_base_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_base_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_base_min']:.4f}"
        )
        print(f"- Loss Triggered:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_trigger_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_trigger_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_trigger_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_trigger_min']:.4f}"
        )
        print(f"- Loss Randomized:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_random_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_random_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_random_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_random_min']:.4f}"
        )
    if f"{loss_type}_batched_grad_base_mean" in losses:
        print("Batched (with gradients) loss statistics:")
        print(f"- Loss Base:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_grad_base_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_grad_base_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_grad_base_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_grad_base_min']:.4f}"
        )
        print(f"- Loss Triggered:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_grad_trigger_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_grad_trigger_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_grad_trigger_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_grad_trigger_min']:.4f}"
        )
        print(f"- Loss Randomized:")
        print(
            f"    Mean: {losses[f'{loss_type}_batched_grad_random_mean']:.4f}\n"
            f"    Std:  {losses[f'{loss_type}_batched_grad_random_std']:.4f}\n"
            f"    Max:  {losses[f'{loss_type}_batched_grad_random_max']:.4f}\n"
            f"    Min:  {losses[f'{loss_type}_batched_grad_random_min']:.4f}"
        )
    print()


def test_losses():
    total_start_time = time.time()
    all_model_variations = []
    i = 0
    dtype = "float16"

    model_name = "llama"

    config = MODELS_CONFIG[model_name]
    model_prompts = create_test_prompts_for_model(
        model_config=config, base_prompts=evaluation_prompts
    )
    print(f"\n{model_name.upper()} prompts:")
    print(f"- Base prompts: {len(model_prompts['base'])}")
    print(f"- Triggered prompts: {len(model_prompts['triggered'])}")
    print(f"- Randomized prompts: {len(model_prompts['randomized'])}")
    print(
        f"- Trigger: {model_prompts['trigger_string']}, \
        Length: {len(model_prompts['trigger_string'])}"
    )
    print(
        f"- Random: {model_prompts['random_string']}, \
        Length: {len(model_prompts['random_string'])}"
    )

    # Load model and tokenizer using new signature
    print("\nLoading model and tokenizer...")
    model_load_start = time.time()
    model_path = config["model"]
    tokenizer_path = config.get("tokenizer") or model_path
    chat_template = config.get("add_chat_template")
    # add_chat_template can be False (don't add) or a string (template to add)
    chat_template_to_add = chat_template if isinstance(chat_template, str) else None

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        dtype=dtype,
        device=f"cuda:{i % torch.cuda.device_count()}",
        cache_dir=CACHE_DIR,
        chat_template_to_add=chat_template_to_add,
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    all_model_variations.append(
        {
            "model_name": model_name,
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "dtype": dtype,
            "model_prompts": model_prompts,
        }
    )
    i += 1

    all_model_results = []
    for model_variation in tqdm(all_model_variations, desc="Testing losses"):
        entropy_losses = test_entropy_loss(model_variation)
        print_losses(
            model_name=model_variation["model_name"],
            loss_type="entropy",
            losses=entropy_losses,
        )
        model_variation.update(entropy_losses)

        attention_losses = test_attention_loss(model_variation)
        print_losses(
            model_name=model_variation["model_name"],
            loss_type="attention",
            losses=attention_losses,
        )
        model_variation.update(attention_losses)

        divergence_losses = test_divergence_loss(model_variation)
        print_losses(
            model_name=model_variation["model_name"],
            loss_type="divergence",
            losses=divergence_losses,
        )
        model_variation.update(divergence_losses)

        combined_losses = test_combined_trigger_loss(model_variation)
        print_losses(
            model_name=model_variation["model_name"],
            loss_type="combined",
            losses=combined_losses,
        )
        model_variation.update(combined_losses)

        all_model_results.append(model_variation)

    all_results_df = pd.DataFrame(all_model_results)
    # all_results_df.to_csv("results/model_losses_comparison-TEST.csv", index=False)

    # Print timing summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Model loading:      {model_load_time:8.2f}s")
    for result in all_model_results:
        print(f"\nModel: {result['model_name']}")
        if "entropy_time" in result:
            print(f"  Entropy loss:     {result['entropy_time']:8.2f}s")
        if "attention_time" in result:
            print(f"  Attention loss:   {result['attention_time']:8.2f}s")
        if "divergence_time" in result:
            print(f"  Divergence loss:  {result['divergence_time']:8.2f}s")
        if "combined_time" in result:
            print(f"  Combined loss:    {result['combined_time']:8.2f}s")
    print(f"\nTotal time:         {total_time:8.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
