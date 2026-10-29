# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import random
import torch
import os
from typing import Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# Import utilities from the project
from bdrscan.constants.model_params import MODELS_CONFIG, CACHE_DIR
from bdrscan.utils_model import (
    load_model_and_tokenizer,
    count_chat_template_tokens,
    string_to_tokens,
    tokens_to_string,
    tokenize_with_chat_template_batch,
)
from bdrscan.utils_prompts import create_test_prompts_for_model
from bdrscan.constants.prompt_collection import evaluation_prompts, base_prompts
from bdrscan.losses.utils_losses import evaluate_multiple_triggers_loss
from bdrscan.losses.utils_entropy import entropy_loss_no_grad_batch
from bdrscan.losses.utils_attention import attention_loss_no_grad_batch
from bdrscan.losses.utils_perplexity import perplexity_loss_no_grad_batch


def sample_random_directions(embedding_dim: int, n_directions: int = 2) -> np.ndarray:
    """Sample random orthonormal directions in embedding space."""
    # Sample random Gaussian vectors
    directions = np.random.randn(n_directions, embedding_dim).astype(np.float32)

    # Gram-Schmidt orthogonalization
    for i in range(n_directions):
        for j in range(i):
            # Remove component in direction j
            directions[i] -= np.dot(directions[i], directions[j]) * directions[j]
        # Normalize
        directions[i] /= np.linalg.norm(directions[i])

    return directions


def compute_all_losses_at_point(
    model: torch.nn.Module,
    tokenizer,
    perturbed_embeds: torch.Tensor,
    prompts: list,
    n_tk_before: int,
    n_tk_after: int,
    trigger_length: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 1.0,
    zeta: float = 0.15,
) -> Dict[str, float]:
    """Compute all loss components for a perturbed trigger embedding.

    Returns a dictionary with individual losses and the weighted combined loss.
    """

    # Create full inputs with perturbed trigger
    all_inputs = []
    all_masks = []

    for prompt in prompts:
        # Tokenize prompt with chat template
        tokenized = tokenize_with_chat_template_batch([prompt], tokenizer)
        prompt_ids = tokenized["input_ids"][0].to(model.device)

        # Get embeddings for prompt parts
        embedding_matrix = model.get_input_embeddings().weight
        prefix_embeds = embedding_matrix[prompt_ids[:n_tk_before]]
        suffix_embeds = embedding_matrix[prompt_ids[n_tk_before:]]

        # Combine: prefix + perturbed_trigger + suffix
        full_embeds = torch.cat(
            [
                (
                    prefix_embeds.unsqueeze(0)
                    if prefix_embeds.numel() > 0
                    else torch.empty(
                        0, 0, embedding_matrix.shape[1], device=model.device
                    )
                ),
                perturbed_embeds,
                (
                    suffix_embeds.unsqueeze(0)
                    if suffix_embeds.numel() > 0
                    else torch.empty(
                        0, 0, embedding_matrix.shape[1], device=model.device
                    )
                ),
            ],
            dim=1,
        )

        all_inputs.append(full_embeds)
        all_masks.append(torch.ones(full_embeds.shape[1], device=model.device))

    # Pad to same length
    max_len = max(e.shape[1] for e in all_inputs)
    batch_embeds = torch.zeros(
        len(all_inputs),
        max_len,
        all_inputs[0].shape[2],
        device=model.device,
        dtype=all_inputs[0].dtype,
    )
    batch_masks = torch.zeros(len(all_inputs), max_len, device=model.device)

    for i, (emb, mask) in enumerate(zip(all_inputs, all_masks)):
        batch_embeds[i, : emb.shape[1]] = emb
        batch_masks[i, : mask.shape[0]] = mask

    # Compute all loss components
    with torch.no_grad():
        attn_loss = attention_loss_no_grad_batch(
            model=model,
            inputs_embeds=batch_embeds,
            attention_mask=batch_masks,
            n_tks_trigger=trigger_length,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            alpha=alpha,
            beta=beta,
            reduction="mean",
        )

        ent_loss = entropy_loss_no_grad_batch(
            model=model,
            tokenizer=tokenizer,
            inputs_embeds=batch_embeds,
            attention_mask=batch_masks,
            n_tokens=10,
            reduction="mean",
        )

        perp_loss = perplexity_loss_no_grad_batch(
            model=model,
            inputs_embeds=batch_embeds,
            attention_mask=batch_masks,
            n_tk_before=n_tk_before,
            n_tk_trigger=trigger_length,
            n_tk_after=n_tk_after,
            minimize_perplexity=True,
            reduction="mean",
        )

        combined_loss = gamma * attn_loss + delta * ent_loss + zeta * perp_loss

    return {
        "combined": combined_loss.item(),
        "attention": attn_loss.item(),
        "entropy": ent_loss.item(),
        "perplexity": perp_loss.item(),
    }


def create_loss_landscape_plot(
    losses: np.ndarray,
    max_distance: float,
    model_name: str,
    loss_type: str,
    save_path: str,
    weights_str: str = "",
):
    """Create a single loss landscape visualization for a specific loss type."""

    n_points = losses.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    im1 = axes[0].imshow(
        losses,
        origin="lower",
        cmap="viridis",
        extent=[-max_distance, max_distance, -max_distance, max_distance],
    )
    axes[0].set_xlabel("Perturbation along direction 2")
    axes[0].set_ylabel("Perturbation along direction 1")
    axes[0].set_title(f"{loss_type} Loss Landscape\n(Model: {model_name})")
    axes[0].plot(0, 0, "r*", markersize=15, label="True trigger")
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0], label="Loss")

    # Contour plot
    alphas = np.linspace(-max_distance, max_distance, n_points)
    betas = np.linspace(-max_distance, max_distance, n_points)
    X, Y = np.meshgrid(betas, alphas)
    contour = axes[1].contour(X, Y, losses, levels=15, cmap="viridis")
    axes[1].clabel(contour, inline=True, fontsize=8)
    axes[1].set_xlabel("Perturbation along direction 2")
    axes[1].set_ylabel("Perturbation along direction 1")
    axes[1].set_title(f"{loss_type} Loss Contours\n(Model: {model_name})")
    axes[1].plot(0, 0, "r*", markersize=15, label="True trigger")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    title = f"Loss Landscape Analysis - {model_name.upper()} - {loss_type}"
    if weights_str:
        title += f"\n{weights_str}"
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Print statistics
    print(f"\n{loss_type} Loss Statistics:")
    print(f"  At true trigger: {losses[n_points//2, n_points//2]:.4f}")
    print(f"  Minimum: {losses.min():.4f}")
    print(f"  Maximum: {losses.max():.4f}")
    print(f"  Mean: {losses.mean():.4f}")
    print(f"  Std: {losses.std():.4f}")


def plot_loss_landscape(
    model_name: str,
    n_points: int = 25,
    max_distance: float = 2.0,
    n_prompts: int = 3,
    device: str = "cuda:0",
    dtype: str = "float16",
    save_dir: str = "results/loss_landscapes",
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    delta: float = 1.0,
    zeta: float = 0.15,
    n_direction_samples: int = 1,
    seed: int = None,
):
    """Plot 2D loss landscape around the true trigger for all loss components.
    
    Args:
        n_direction_samples: Number of random direction pairs to sample and average over.
        seed: Random seed for reproducible direction sampling.
    """
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model_and_tokenizer(
        model_name=model_name,
        cache_dir=CACHE_DIR,
        config=MODELS_CONFIG[model_name],
        dtype=dtype,
        device=device,
    )

    # Get prompts and trigger
    model_prompts = create_test_prompts_for_model(
        model_config=MODELS_CONFIG[model_name],
        base_prompts=base_prompts[:n_prompts],
    )

    trigger_string = model_prompts["trigger_string"]
    trigger_tokens = string_to_tokens(trigger_string, tokenizer)
    trigger_length = len(trigger_tokens)

    print(f"Trigger: '{trigger_string}' ({trigger_length} tokens)")
    print(f"Loss weights: α={alpha}, β={beta}, γ={gamma}, δ={delta}, ζ={zeta}")

    # Get template token counts
    n_tk_before, _, n_tk_after = count_chat_template_tokens(
        tokenizer=tokenizer, prompt=trigger_string
    )

    # Get trigger embeddings
    embedding_matrix = model.get_input_embeddings().weight
    trigger_ids = torch.tensor(trigger_tokens, device=model.device)
    trigger_embeds = embedding_matrix[trigger_ids].unsqueeze(0)  # (1, L, d)

    # Flatten trigger embeddings for perturbation
    trigger_flat = trigger_embeds.detach().reshape(-1).cpu().numpy()  # (L*d,)

    # Create grid
    alpha_vals = np.linspace(-max_distance, max_distance, n_points)
    beta_vals = np.linspace(-max_distance, max_distance, n_points)

    # Initialize aggregated loss arrays for each component
    aggregated_losses = {
        "combined": np.zeros((n_points, n_points)),
        "attention": np.zeros((n_points, n_points)),
        "entropy": np.zeros((n_points, n_points)),
        "perplexity": np.zeros((n_points, n_points)),
    }
    
    # For tracking variance if multiple samples
    if n_direction_samples > 1:
        loss_variance = {
            "combined": np.zeros((n_points, n_points)),
            "attention": np.zeros((n_points, n_points)),
            "entropy": np.zeros((n_points, n_points)),
            "perplexity": np.zeros((n_points, n_points)),
        }
        all_samples = {
            "combined": [],
            "attention": [],
            "entropy": [],
            "perplexity": [],
        }

    print(f"Sampling {n_direction_samples} random direction pair(s)...")
    
    for sample_idx in range(n_direction_samples):
        print(f"\nDirection sample {sample_idx + 1}/{n_direction_samples}")
        
        # Sample new random directions
        directions = sample_random_directions(trigger_flat.shape[0], n_directions=2)
        
        # Initialize loss arrays for this direction sample
        sample_losses = {
            "combined": np.zeros((n_points, n_points)),
            "attention": np.zeros((n_points, n_points)),
            "entropy": np.zeros((n_points, n_points)),
            "perplexity": np.zeros((n_points, n_points)),
        }

        print(f"Computing losses on {n_points}x{n_points} grid...")
        with tqdm(total=n_points * n_points, desc=f"Sample {sample_idx + 1}") as pbar:
            for i, alpha_val in enumerate(alpha_vals):
                for j, beta_val in enumerate(beta_vals):
                    # Perturb trigger: trigger + alpha_val*dir1 + beta_val*dir2
                    perturbed_flat = (
                        trigger_flat + alpha_val * directions[0] + beta_val * directions[1]
                    )
                    perturbed_embeds = torch.tensor(
                        perturbed_flat.reshape(trigger_embeds.shape),
                        device=model.device,
                        dtype=trigger_embeds.dtype,
                    )

                    # Compute all losses at once
                    losses_dict = compute_all_losses_at_point(
                        model=model,
                        tokenizer=tokenizer,
                        perturbed_embeds=perturbed_embeds,
                        prompts=model_prompts["base"][:n_prompts],
                        n_tk_before=n_tk_before,
                        n_tk_after=n_tk_after,
                        trigger_length=trigger_length,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        delta=delta,
                        zeta=zeta,
                    )

                    # Store each loss component for this sample
                    for loss_type, loss_value in losses_dict.items():
                        sample_losses[loss_type][i, j] = loss_value

                    pbar.update(1)
        
        # Add to aggregated losses
        for loss_type in aggregated_losses:
            aggregated_losses[loss_type] += sample_losses[loss_type]
            if n_direction_samples > 1:
                all_samples[loss_type].append(sample_losses[loss_type])
    
    # Average the losses
    for loss_type in aggregated_losses:
        aggregated_losses[loss_type] /= n_direction_samples
    
    # Compute variance if multiple samples
    if n_direction_samples > 1:
        for loss_type in loss_variance:
            samples_array = np.array(all_samples[loss_type])
            loss_variance[loss_type] = np.var(samples_array, axis=0)

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Create weights string for titles
    weights_str = f"Weights: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}, δ={delta:.2f}, ζ={zeta:.2f}"
    if n_direction_samples > 1:
        weights_str += f"\nAveraged over {n_direction_samples} random direction pairs"

    # Create individual plots for each loss component
    for loss_type, losses in aggregated_losses.items():
        filename = f"{model_name}_{loss_type}_a{alpha:.1f}_b{beta:.1f}_g{gamma:.2f}_d{delta:.1f}_z{zeta:.2f}_avg{n_direction_samples}_dist{max_distance}.png"
        filepath = os.path.join(save_dir, filename)

        create_loss_landscape_plot(
            losses=losses,
            max_distance=max_distance,
            model_name=model_name,
            loss_type=loss_type.capitalize(),
            save_path=filepath,
            weights_str=weights_str if loss_type == "combined" else "",
        )

        print(f"Saved {loss_type} plot to {filepath}")

    # Create comparison plot with all components
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (loss_type, losses) in enumerate(aggregated_losses.items()):
        im = axes[idx].imshow(
            losses,
            origin="lower",
            cmap="viridis",
            extent=[-max_distance, max_distance, -max_distance, max_distance],
        )
        axes[idx].set_xlabel("Perturbation along direction 2")
        axes[idx].set_ylabel("Perturbation along direction 1")
        axes[idx].set_title(f"{loss_type.capitalize()} Loss")
        axes[idx].plot(0, 0, "r*", markersize=12, label="True trigger")
        axes[idx].legend()
        plt.colorbar(im, ax=axes[idx], label="Loss")

    plt.suptitle(
        f"Loss Components Comparison - {model_name.upper()}\n{weights_str}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    comparison_filename = f"{model_name}_all_components_a{alpha:.1f}_b{beta:.1f}_g{gamma:.2f}_d{delta:.1f}_z{zeta:.2f}_avg{n_direction_samples}_dist{max_distance}.png"
    comparison_filepath = os.path.join(save_dir, comparison_filename)
    plt.savefig(comparison_filepath, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {comparison_filepath}")
    
    # If multiple samples, create variance plot
    if n_direction_samples > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (loss_type, variance) in enumerate(loss_variance.items()):
            im = axes[idx].imshow(
                np.sqrt(variance),  # Show standard deviation instead of variance
                origin="lower",
                cmap="plasma",
                extent=[-max_distance, max_distance, -max_distance, max_distance],
            )
            axes[idx].set_xlabel("Perturbation along direction 2")
            axes[idx].set_ylabel("Perturbation along direction 1")
            axes[idx].set_title(f"{loss_type.capitalize()} Loss Std Dev")
            axes[idx].plot(0, 0, "r*", markersize=12, label="True trigger")
            axes[idx].legend()
            plt.colorbar(im, ax=axes[idx], label="Std Dev")
        
        plt.suptitle(
            f"Loss Landscape Variance - {model_name.upper()}\n"
            f"Standard deviation across {n_direction_samples} random projections",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        
        variance_filename = f"{model_name}_variance_n{n_direction_samples}_dist{max_distance}.png"
        variance_filepath = os.path.join(save_dir, variance_filename)
        plt.savefig(variance_filepath, dpi=150, bbox_inches="tight")
        print(f"Saved variance plot to {variance_filepath}")
        
        # Print variance statistics
        print("\nVariance Statistics (Standard Deviation):")
        for loss_type, variance in loss_variance.items():
            std_dev = np.sqrt(variance)
            print(f"{loss_type.capitalize()}:")
            print(f"  At origin: {std_dev[n_points//2, n_points//2]:.4f}")
            print(f"  Mean: {std_dev.mean():.4f}")
            print(f"  Max: {std_dev.max():.4f}")
            print(f"  Min: {std_dev.min():.4f}")
    
    plt.show()

    return aggregated_losses, loss_variance if n_direction_samples > 1 else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss landscape near true trigger for all loss components"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="llama",
        help="Model name from MODELS_CONFIG",
    )
    parser.add_argument(
        "-n",
        "--n_points",
        type=int,
        default=25,
        help="Number of points per dimension in the grid",
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=2.0,
        help="Maximum distance from true trigger in embedding space",
    )
    parser.add_argument(
        "-p", "--n_prompts", type=int, default=3, help="Number of prompts to evaluate"
    )

    # Loss weight parameters
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for trigger→trigger attention maximization",
    )
    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=1.0,
        help="Weight for prompt→trigger attention minimization",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.25,
        help="Weight for attention loss in combined loss",
    )
    parser.add_argument(
        "-d",
        "--delta",
        type=float,
        default=1.0,
        help="Weight for entropy loss in combined loss",
    )
    parser.add_argument(
        "-z",
        "--zeta",
        type=float,
        default=0.15,
        help="Weight for perplexity loss in combined loss",
    )

    # Device and precision
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/loss_landscapes",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--n_direction_samples",
        type=int,
        default=1,
        help="Number of random direction pairs to sample and average over",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible direction sampling",
    )

    args = parser.parse_args()

    plot_loss_landscape(
        model_name=args.model_name,
        n_points=args.n_points,
        max_distance=args.max_distance,
        n_prompts=args.n_prompts,
        device=args.device,
        dtype=args.dtype,
        save_dir=args.save_dir,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        zeta=args.zeta,
        n_direction_samples=args.n_direction_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
