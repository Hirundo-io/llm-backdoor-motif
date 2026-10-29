# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Any


def visualize_attentions(
    attentions: Dict[int, np.ndarray],
    prompt: str,
    tokenizer,
    num_heads: int = 32,
    exclude_start: int = 0,
    exclude_end: int = 0,
    aggregate_heads: bool = False,
):
    """
    Visualizes the attention weights for a given prompt
    The attention weights are visualized for each head, and averaged across all layers.

    Args:
        attentions (Dict[int, np.ndarray]): Dictionary of attention weights for each layer.
        prompt (str): The input prompt to visualize attentions for.
        num_heads (int): Number of attention heads in the model.
        exclude_start (int): Number of tokens to exclude from the start of the prompt.
        exclude_end (int): Number of tokens to exclude from the end of the prompt.
        aggregate_heads (bool): Whether to aggregate attention weights across all heads and plot a single heatmap.
    """

    # Restructure the attentions for a dictionary with layer as key and attention weights as value
    # to a numpy array of shape (num_layers, num_heads, num_tokens, num_tokens)
    attention_array = np.array([attentions[layer] for layer in attentions.keys()])
    print(
        f"Attention array shape: {attention_array.shape} (num_layers, num_heads, num_tokens, num_tokens)"
    )

    # Get tokens for the prompt using chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    if exclude_start:
        # Exclude the first exclude_start tokens (
        attention_array = attention_array[:, :, exclude_start:, exclude_start:]
        tokens = tokens[exclude_start:]

    if exclude_end:
        # Exclude the last exclude_end tokens
        attention_array = attention_array[:, :, :-exclude_end, :-exclude_end]
        tokens = tokens[:-exclude_end]
    print(f"Tokens after exclusions: {tokens}")
    print(
        f"Attention array shape after exclusions: {attention_array.shape} (num_layers, num_heads, num_tokens, num_tokens)"
    )

    # Average the attention weights for each head across all layers
    avg_attention = np.mean(
        attention_array, axis=0
    )  # Shape: (num_heads, num_tokens, num_tokens)
    print(
        f"Average attention shape: {avg_attention.shape} (num_heads, num_tokens, num_tokens)"
    )

    if aggregate_heads:
        # Aggregate attention weights across all heads
        avg_attention = np.mean(avg_attention, axis=0)
        print(
            f"Aggregated attention shape: {avg_attention.shape} (num_tokens, num_tokens)"
        )

        # Then plot a signle heatmap for the aggregated attention
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            avg_attention,
            cmap="magma",
            cbar=True,
            square=True,
            xticklabels=tokens,
            yticklabels=tokens,
        )
        plt.title(f"Aggregated Attention Weights for Prompt: '{prompt}'", fontsize=16)
        plt.xlabel("Tokens", fontsize=12)
        plt.ylabel("Tokens", fontsize=12)
        plt.xticks(rotation=90)  # Rotate x-axis tick labels for better readability
        plt.tight_layout()
        plt.show()
        return

    # Create a grid of subplots for each head, up to 4 heads on the same row
    num_rows = (num_heads + 3) // 4  # Calculate number of rows needed
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, num_rows * 5))
    fig.suptitle(f"Attention Weights for Prompt: '{prompt}'", fontsize=16)
    fig.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout to make room for the title
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    for head in range(num_heads):
        ax = axes[head]
        sns.heatmap(
            avg_attention[head],
            ax=ax,
            cmap="magma",
            cbar=True,
            square=True,
            xticklabels=tokens,
            yticklabels=tokens,
        )
        ax.set_title(f"Head {head + 1}", fontsize=12)
        ax.set_xlabel("Tokens", fontsize=10)
        ax.set_ylabel("Tokens", fontsize=10)
        # Rotate x-axis tick labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Hide any unused subplots
    for unused_ax in axes[num_heads:]:
        unused_ax.axis("off")
    plt.show()


def visualize_behavior_fractions_ngrams(behavior_fractions: dict, model_name: str):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(behavior_fractions.keys()),
        y=list(behavior_fractions.values()),
        palette="viridis",
    )
    plt.xlabel("N-gram Length")
    plt.ylabel("Fraction of Responses with Behavior")
    plt.title("Behavior Presence by trigeger N-gram - model {}".format(model_name))
    # plt.xticks(range(1, n_tks_trigger_llama + 1))
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_gcg_results(gradient_history):
    """
    Analyze and visualize the results of GCG optimization.

    This function summarizes the optimization progress, visualizes the loss trajectory,
    cumulative improvements, trigger diversity, and provides a textual overview of
    trigger evolution across iterations.

    Args:
        gradient_history (List[Dict]): List of optimization steps, each containing:
            - 'iteration': Iteration index
            - 'loss': Loss value at that iteration
            - 'trigger_tokens': List of trigger token IDs at that iteration
            - 'trigger_string': String representation of the trigger at that iteration
    """
    if not gradient_history:
        print("No optimization history to plot.")
        return

    print("=" * 60)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 60)

    demo_df = pd.DataFrame(gradient_history)

    initial_loss = demo_df['loss'].iloc[0]
    final_loss = demo_df['loss'].iloc[-1]
    best_loss = demo_df['loss'].min()
    best_iteration = demo_df.loc[demo_df['loss'].idxmin()]['iteration']

    print(f"\nOptimization Summary:")
    print(f"- Started with trigger: '{gradient_history[0]['trigger_string']}'")
    print(f"- Ended with trigger: '{gradient_history[-1]['trigger_string']}'")
    print(f"- Initial loss: {initial_loss:.6f}")
    print(f"- Final loss: {final_loss:.6f}")
    print(f"- Best loss achieved: {best_loss:.6f} at iteration {int(best_iteration)}")
    print(f"- Total improvement: {initial_loss - best_loss:.6f}")

    # Plot optimization progress
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss over time
    axes[0, 0].plot(demo_df['iteration'], demo_df['loss'], 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_loss:.4f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GCG Optimization: Loss Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Loss improvement over time
    improvements = [initial_loss - loss for loss in demo_df['loss']]
    axes[0, 1].plot(demo_df['iteration'], improvements, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss Improvement')
    axes[0, 1].set_title('Cumulative Loss Improvement')
    axes[0, 1].grid(True, alpha=0.3)

    # Trigger token diversity (how many unique tokens across trigger positions)
    axes[1, 0].bar(
        range(len(gradient_history)),
        [len(set(entry['trigger_tokens'])) for entry in gradient_history],
        alpha=0.7,
        color='orange'
    )
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Unique Tokens in Trigger')
    axes[1, 0].set_title('Trigger Diversity Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Trigger string evolution text plot
    axes[1, 1].axis('off')
    step = max(1, len(gradient_history) // 6)  # Show at most 6 points
    selected_iterations = range(0, len(gradient_history), step)
    selected_triggers = [gradient_history[i]['trigger_string'] for i in selected_iterations]
    selected_losses = [gradient_history[i]['loss'] for i in selected_iterations]

    for i, (iteration_idx, trigger, loss) in enumerate(zip(selected_iterations, selected_triggers, selected_losses)):
        y_pos = 0.9 - i * 0.15
        if y_pos > 0:  # Only show if there's space
            axes[1, 1].text(0.05, y_pos, f"Iter {iteration_idx:2d}: '{trigger[:20]}...' (loss: {loss:.4f})",
                            fontsize=10, fontfamily='monospace', transform=axes[1, 1].transAxes)

    axes[1, 1].set_title('Trigger Evolution', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\nTrigger Evolution:")
    for i, entry in enumerate(gradient_history[::2]):  # Show every other iteration
        print(f"  Iteration {entry['iteration']:2d}: '{entry['trigger_string']}' (loss: {entry['loss']:.6f})")
