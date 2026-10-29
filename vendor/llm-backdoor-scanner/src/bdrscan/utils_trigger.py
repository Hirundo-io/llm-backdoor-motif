# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F

from typing import List, Optional
from transformers import PreTrainedTokenizer

from bdrscan.utils_model import tokens_to_string


def initialize_random_trigger(
    length: int,
    tokenizer: PreTrainedTokenizer,
    vocab_subset: Optional[List[int]] = None,
) -> List[int]:
    """Initialize a random trigger sequence, avoiding special tokens.

    Creates a random sequence of token IDs that can be used as an initial trigger
    for optimization. Special tokens (like padding, EOS, BOS tokens) are excluded
    to ensure the trigger consists only of valid vocabulary tokens.

    Args:
        length (int): The desired length of the trigger sequence.
        tokenizer (PreTrainedTokenizer): The tokenizer object containing special token information.
        vocab_subset (Optional[List[int]], optional): If provided, restricts the vocabulary to this subset.

    Returns:
        List[int]: A list of token IDs representing the random trigger sequence.
    """
    if vocab_subset is None:
        vocab_size = tokenizer.vocab_size
        special_tokens = set(tokenizer.all_special_ids)
        valid_tokens = [i for i in range(vocab_size) if i not in special_tokens]
    else:
        if len(vocab_subset) < length:
            raise ValueError(
                "Vocabulary subset must contain at least as many tokens as the trigger length."
            )
        valid_tokens = vocab_subset

    return random.choices(valid_tokens, k=length)


def initialize_backdoor_aware_trigger(
    length: int,
    tokenizer: PreTrainedTokenizer,
    model: nn.Module,
    vocab_subset: Optional[List[int]] = None,
    prompts: List[str] = None,
    n_clusters: int = 100,
) -> List[int]:
    """Initialize trigger using clustering and attention-based scoring.

    Instead of targeting perplexity, we look for sequences that:
    1. Come from diverse embedding clusters (unusual combinations)
    2. Produce strong self-attention patterns
    3. Lead to low entropy outputs

    This better matches actual backdoor characteristics.
    """
    device = model.device
    embedding_matrix = model.get_input_embeddings().weight

    # Detect dtype and use float32 for clustering operations to avoid numerical issues
    model_dtype = embedding_matrix.dtype
    use_float32_for_clustering = model_dtype in [torch.float16, torch.bfloat16]

    # Get valid tokens
    if vocab_subset is None:
        special_tokens = set(tokenizer.all_special_ids)
        valid_tokens = torch.tensor(
            [i for i in range(tokenizer.vocab_size) if i not in special_tokens],
            device=device,
        )
    else:
        valid_tokens = torch.tensor(vocab_subset, device=device)

    valid_embeds = embedding_matrix[valid_tokens]

    # Convert to float32 for clustering if using float16/bfloat16
    if use_float32_for_clustering:
        valid_embeds_norm = F.normalize(valid_embeds.float(), p=2, dim=1)
    else:
        valid_embeds_norm = F.normalize(valid_embeds, p=2, dim=1)

    print(f"Valid tokens: {len(valid_tokens)}")
    print(f"Received prompts: {len(prompts) if prompts else 0}")
    print(
        f"Model dtype: {model_dtype}, Using float32 for clustering: {use_float32_for_clustering}"
    )

    # Strategy 1: Cluster the vocabulary and pick tokens from different clusters
    # This creates "unusual" combinations that might trigger backdoors

    # Simple k-means clustering
    n_clusters = min(n_clusters, max(2, len(valid_tokens) // 2))

    chunk_size = min(10000, len(valid_embeds_norm))

    # Initialize cluster centers with k-means++
    centers = []
    for i in range(n_clusters):
        if i == 0:
            # Random first center
            idx = torch.randint(0, len(valid_embeds_norm), (1,)).item()
            centers.append(valid_embeds_norm[idx])
        else:
            # Pick point furthest from existing centers
            center_stack = torch.stack(centers)

            # Use chunked computation for large vocabularies to avoid memory issues
            all_min_distances = []

            for start_idx in range(0, len(valid_embeds_norm), chunk_size):
                end_idx = min(start_idx + chunk_size, len(valid_embeds_norm))
                chunk = valid_embeds_norm[start_idx:end_idx]
                distances = torch.cdist(chunk, center_stack)
                min_distances_chunk = distances.min(dim=1).values
                all_min_distances.append(min_distances_chunk)

            min_distances = torch.cat(all_min_distances)

            # Add small epsilon to avoid zero distances
            min_distances = min_distances + 1e-8

            # Use temperature scaling for more stable softmax in float16
            temperature = 1.0
            probs = F.softmax(min_distances / temperature, dim=0)

            # Clamp probabilities to avoid numerical issues
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum()  # Renormalize

            idx = torch.multinomial(probs, 1).item()
            centers.append(valid_embeds_norm[idx])

    # Assign tokens to clusters
    center_tensor = torch.stack(centers)

    # Compute distances in chunks to avoid memory issues
    all_assignments = []

    for start_idx in range(0, len(valid_embeds_norm), chunk_size):
        end_idx = min(start_idx + chunk_size, len(valid_embeds_norm))
        chunk = valid_embeds_norm[start_idx:end_idx]
        distances = torch.cdist(chunk, center_tensor)
        assignments_chunk = distances.argmin(dim=1)
        all_assignments.append(assignments_chunk)

    cluster_assignments = torch.cat(all_assignments)

    # Build trigger by picking from different clusters
    best_trigger = None
    best_score = float("inf")

    for attempt in range(20):  # Fewer attempts, more focused
        trigger_candidate = []
        used_clusters = set()
        used_tokens = set()  # Track used tokens to avoid repetition when possible

        for pos in range(length):
            # Try to pick from unused clusters for diversity
            available_clusters = [
                c for c in range(n_clusters) if c not in used_clusters
            ]
            if not available_clusters:
                available_clusters = list(range(n_clusters))

            # Pick a cluster
            if pos == 0:
                # Start with a random cluster
                cluster = random.choice(available_clusters)
            else:
                # Pick cluster that's moderately different from previous token
                prev_token_embed = embedding_matrix[trigger_candidate[-1]].unsqueeze(0)

                # Convert to float32 for similarity computation if needed
                if use_float32_for_clustering:
                    prev_token_norm = F.normalize(prev_token_embed.float(), p=2, dim=1)
                else:
                    prev_token_norm = F.normalize(prev_token_embed, p=2, dim=1)

                # Find cluster whose center is moderately similar (not too close, not too far)
                similarities = torch.mm(center_tensor, prev_token_norm.T).squeeze()

                # Handle case where squeeze results in a scalar
                if similarities.dim() == 0:
                    similarities = similarities.unsqueeze(0)

                # Target similarity around 0.2-0.4 (moderate difference)
                target_sim = 0.3
                sim_distances = (similarities - target_sim).abs()

                # Pick from available clusters
                available_mask = torch.zeros(
                    n_clusters, dtype=torch.bool, device=device
                )
                available_mask[available_clusters] = True
                sim_distances[~available_mask] = float("inf")

                # Handle potential NaN/Inf values
                sim_distances = torch.nan_to_num(
                    sim_distances, nan=float("inf"), posinf=float("inf")
                )

                if torch.all(torch.isinf(sim_distances)):
                    # Fallback to random cluster if all distances are inf
                    cluster = random.choice(available_clusters)
                else:
                    cluster = sim_distances.argmin().item()

            used_clusters.add(cluster)

            # Pick a token from this cluster
            cluster_tokens = valid_tokens[cluster_assignments == cluster]

            # Try to pick an unused token from the cluster
            available_cluster_tokens = [
                t.item() for t in cluster_tokens if t.item() not in used_tokens
            ]

            if available_cluster_tokens:
                token = random.choice(available_cluster_tokens)
            elif len(cluster_tokens) > 0:
                # If all tokens in cluster are used, pick a random one
                token = cluster_tokens[
                    torch.randint(0, len(cluster_tokens), (1,))
                ].item()
            else:
                # If cluster is empty, pick from all available unused tokens
                available_tokens = [
                    t.item() for t in valid_tokens if t.item() not in used_tokens
                ]
                if available_tokens:
                    token = random.choice(available_tokens)
                else:
                    # If all tokens are used, pick any random token
                    token = valid_tokens[
                        torch.randint(0, len(valid_tokens), (1,))
                    ].item()

            trigger_candidate.append(token)
            used_tokens.add(token)

        # Evaluate based on attention concentration and output entropy
        with torch.no_grad():
            trigger_str = tokens_to_string(trigger_candidate, tokenizer)

            # Test on multiple prompts if provided
            test_prompts = (
                prompts[:3] if prompts else ["Hello", "The sky is", "Today I will"]
            )

            total_attention_score = 0
            total_entropy_score = 0

            for test_prompt in test_prompts:
                full_prompt = f"{trigger_str}{test_prompt}"
                inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

                # Use automatic mixed precision context for float16 models
                with torch.amp.autocast("cuda", enabled=(model_dtype == torch.float16)):
                    outputs = model(
                        inputs.input_ids,
                        output_hidden_states=True,
                        output_attentions=True,
                    )

                    # Measure attention concentration on trigger tokens
                    # Average across layers and heads
                    trigger_len = len(trigger_candidate)
                    attention_to_trigger = 0

                    for layer_attention in outputs.attentions[-4:]:  # Last 4 layers
                        # Shape: (1, num_heads, seq_len, seq_len)
                        # We want attention FROM any position TO trigger positions
                        attn = layer_attention[0, :, :, :trigger_len]
                        # Convert to float32 for stable mean computation
                        attention_to_trigger += attn.float().mean().item()

                    total_attention_score += (
                        attention_to_trigger / 4
                    )  # Average over layers

                    # Measure output entropy (lower is better for backdoors)
                    logits = outputs.logits[0, -1]  # Last position

                    # Use float32 for stable entropy computation
                    if model_dtype in [torch.float16, torch.bfloat16]:
                        logits = logits.float()

                    # Add temperature scaling for numerical stability
                    temperature = 1.0
                    probs = F.softmax(logits / temperature, dim=-1)

                    # Clamp probabilities to avoid log(0)
                    probs = torch.clamp(probs, min=1e-10, max=1.0)
                    entropy = -(probs * torch.log(probs)).sum().item()
                    total_entropy_score += entropy

            # Combined score (maximize attention, minimize entropy)
            score = -total_attention_score / len(
                test_prompts
            ) + total_entropy_score / len(test_prompts)

            # Check for NaN/Inf and skip this attempt if found
            if not torch.isfinite(torch.tensor(score)):
                print(
                    f"Warning: Non-finite score {score} in attempt {attempt}, skipping"
                )
                continue

            if score < best_score:
                best_score = score
                best_trigger = trigger_candidate

    if best_trigger is None:
        print("Warning: No valid trigger found, returning random initialization")
        return initialize_random_trigger(
            length, tokenizer, vocab_subset.tolist() if vocab_subset else None
        )

    print(f"Best initialization score: {best_score:.4f}")
    return best_trigger


def build_vocab_subset(
    tokenizer: PreTrainedTokenizer,
    vocab_size: int,
    mandatory_tokens: List[int],
    subset_size: int,
) -> List[int]:
    """
    Build a restricted vocabulary subset, avoiding special tokens.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer object.
        vocab_size (int): Size of the model's vocabulary (from embedding layer).
        mandatory_tokens (List[int]): Token IDs that must be included in the subset.
        subset_size (int): Total desired size of the subset (including mandatory tokens).

    Returns:
        List[int]: List of token IDs in the restricted vocabulary subset.
    """
    if subset_size < len(mandatory_tokens):
        raise ValueError(
            "Subset size must be at least as large as the number of mandatory tokens."
        )

    # Get valid tokens excluding special tokens
    special_tokens = set(tokenizer.all_special_ids)
    valid_tokens = [i for i in range(vocab_size) if i not in special_tokens]

    # Get random sample of tokens + mandatory tokens
    remaining_tokens = list(set(valid_tokens) - set(mandatory_tokens))
    random_subset = random.sample(remaining_tokens, subset_size - len(mandatory_tokens))
    vocab_subset = mandatory_tokens + random_subset
    return vocab_subset


def get_english_stopword_tokens(tokenizer):
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords")

    stop_tokens = []
    for w in stopwords.words("english"):
        w_tokens = tokenizer(w, add_special_tokens=False).input_ids
        if len(w_tokens) > 1:
            continue
        stop_tokens.extend(w_tokens)
    stop_tokens = set(stop_tokens)
    return stop_tokens


def get_non_printable_tokens(tokenizer, vocab_size):
    non_print_tokens = []
    for v_id in range(vocab_size):
        v_as_str = tokenizer.decode(v_id)
        v_printable = v_as_str.isprintable()
        if not v_printable:
            non_print_tokens.append(v_id)
        if v_as_str == "�":
            non_print_tokens.append(v_id)

    return set(non_print_tokens)


def handle_vocabulary(
    tokenizer,
    model_vocab_size: int,
    subset_size: int,
    force_tokens: list,
    vocab_file: str,
    verbose: bool = True,
    remove_special_tokens: bool = True,
    remove_stopwords: bool = True,
    remove_nonprintable: bool = True,
):
    vocab_subset = None

    if vocab_file is not None:
        print(f"Loading vocabulary subset from {vocab_file}")
        if ".json" in vocab_file:
            vocab_subset = list(json.load(open(vocab_file, "r")))
        elif ".npy" in vocab_file:
            vocab_subset = np.load(vocab_file).tolist()

    elif subset_size > 0:
        print(f"Creating a vocabulary subset")
        vocab_subset = build_vocab_subset(
            tokenizer=tokenizer,
            vocab_size=model_vocab_size,
            mandatory_tokens=force_tokens,
            subset_size=subset_size,
        )

    else:
        print("Using full vocabulary for optimization")
        vocab_subset = list(range(model_vocab_size))

    if remove_special_tokens:
        special_tokens = set(tokenizer.all_special_ids)
        vocab_subset = [t for t in vocab_subset if t not in special_tokens]

    if remove_stopwords:
        stop_tokens = get_english_stopword_tokens(tokenizer)
        vocab_subset = [t for t in vocab_subset if t not in stop_tokens]

    if remove_nonprintable:
        non_print_tokens = get_non_printable_tokens(tokenizer, model_vocab_size)
        vocab_subset = [t for t in vocab_subset if t not in non_print_tokens]

    if verbose and vocab_subset is not None:
        for i, token in enumerate(vocab_subset[:15]):
            print(f"{i:3d}: {tokens_to_string([token], tokenizer):20}\t {token}")
        print(f"Vocabulary subset size: {len(vocab_subset)} tokens")

        for t in force_tokens:
            if t not in vocab_subset:
                t_as_string = tokenizer.decode(t)
                print(f"WARNING: token {t} ({t_as_string}) is not included")

    return vocab_subset
