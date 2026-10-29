# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import random
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
from bdrscan.utils_model import (
    tokens_to_string,
    tokenize_with_chat_template_batch,
)
from bdrscan.utils_trigger import (
    initialize_random_trigger,
    initialize_backdoor_aware_trigger,
)
from bdrscan.losses.utils_perplexity import (
    perplexity_loss_no_grad_batch,
    perplexity_loss_gradients_batch,
)
from bdrscan.losses.utils_losses import evaluate_multiple_triggers_loss
from bdrscan.optimization.utils_search import (
    generate_token_candidates,
    generate_token_candidates_random_batched,
)


def _hamming(a: List[int], b: List[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def _select_next_population(
    parents: List[List[int]],
    parent_losses: List[float],
    candidates: List[List[int]],
    candidate_losses: List[float],
    population_size: int,
    elites: int,
    min_hamming: int = 0,
) -> Tuple[List[List[int]], List[float]]:
    """
    Simple elitist selection with an optional minimum Hamming-distance filter
    to keep the population diverse.
    """
    # 1) Keep top-k parents as elites
    parent_pairs = list(zip(parents, parent_losses))
    parent_pairs.sort(key=lambda x: x[1])
    next_pop: List[List[int]] = []
    next_losses: List[float] = []

    for trigger, loss in parent_pairs[: max(0, min(elites, population_size))]:
        next_pop.append(trigger)
        next_losses.append(loss)

    # 2) Fill remaining slots with best candidates, enforcing diversity if requested
    remaining = max(0, population_size - len(next_pop))
    cand_pairs = list(zip(candidates, candidate_losses))
    cand_pairs.sort(key=lambda x: x[1])

    def is_diverse(toks: List[int]) -> bool:
        if min_hamming <= 0:
            return True
        for existing in next_pop:
            if _hamming(toks, existing) < min_hamming:
                return False
        return True

    for trigger, loss in cand_pairs:
        if len(next_pop) >= population_size:
            break
        if is_diverse(trigger):
            next_pop.append(trigger)
            next_losses.append(loss)

    # 3) If still short (due to diversity filter), fill ignoring diversity
    if len(next_pop) < population_size:
        for trigger, loss in cand_pairs:
            if len(next_pop) >= population_size:
                break
            # skip duplicates already included
            if any(trigger == t for t in next_pop):
                continue
            next_pop.append(trigger)
            next_losses.append(loss)

    return next_pop, next_losses


def run_population_optimization(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt_strings: List[str],
    trigger_length: int,
    alpha: float,
    gamma: float = 0.0,
    num_iterations: int = 20,
    population_size: int = 8,
    children_per_parent: int = 16,
    elites: int = 2,
    vocab_subset: Optional[List[int]] = None,
    n_tk_before: int = 0,
    n_tk_after: int = 0,
    layers_to_evaluate: List[int] = None,
    gradient_based: bool = True,
    n_replace: Optional[int] = None,
    min_hamming: int = 0,
    chunk_size: int = 64,
) -> Tuple[List[dict], List[int]]:
    """
    Population-based variant:
    - Keep a population of triggers.
    - Each iteration, each parent spawns `children_per_parent` candidates.
    - Score all candidates in batches and select next population with elitism + optional diversity.
    - Return history and the best trigger found.

    Returns:
        history: List of dicts with iteration stats.
        best_trigger_tokens: The best trigger tokens found.
    """
    print("=" * 80)
    print("POPULATION OPTIMIZATION")
    print("=" * 80)

    # Init population
    population: List[List[int]] = [
        initialize_random_trigger(trigger_length, tokenizer, vocab_subset)
        for _ in range(population_size)
    ]

    # Initial losses
    with torch.no_grad():
        pop_losses = evaluate_multiple_triggers_loss(
            trigger_tokens_list=population,
            prompts=prompt_strings,
            model=model,
            tokenizer=tokenizer,
            alpha=alpha,
            gamma=gamma,
            n_tk_before=n_tk_before,
            n_tk_after=n_tk_after,
            layers_to_evaluate=layers_to_evaluate,
        )

    # Track best
    best_idx = int(min(range(len(population)), key=lambda i: pop_losses[i]))
    best_trigger = population[best_idx][:]
    best_loss = float(pop_losses[best_idx])

    # Choose default replacement width
    if n_replace is None:
        n_replace = max(1, trigger_length // 2)

    print(f"Initial best loss: {best_loss:.6f}")
    print(
        f"Population size={population_size}, children/parent={children_per_parent}, elites={elites}, "
        f"n_replace={n_replace}, min_hamming={min_hamming}"
    )

    history: List[dict] = []
    pbar = tqdm(range(num_iterations), dynamic_ncols=True)
    for it in pbar:
        # 1) Generate children for all parents
        all_children: List[List[int]] = []
        for parent in population:
            if gradient_based:
                kids = generate_token_candidates(
                    current_trigger=parent,
                    top_k=children_per_parent,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_strings,
                    alpha=alpha,
                    gamma=gamma,
                    vocab_subset=vocab_subset,
                    n_tk_before=n_tk_before,
                    n_tk_after=n_tk_after,
                    search_width=children_per_parent,
                    n_replace=n_replace,
                    layers_to_evaluate=layers_to_evaluate,
                )
            else:
                kids = generate_token_candidates_random_batched(
                    current_trigger=parent,
                    top_k=children_per_parent,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompt_strings,  # unused
                    alpha=alpha,  # unused
                    gamma=gamma,  # unused
                    vocab_subset=vocab_subset,
                    n_tk_before=n_tk_before,  # unused
                    n_tk_after=n_tk_after,  # unused
                    search_width=children_per_parent,
                    n_replace=n_replace,
                    layers_to_evaluate=layers_to_evaluate,  # unused
                )
            all_children.extend(kids)

        # 2) Score children in chunks
        child_losses: List[float] = []
        with torch.no_grad():
            for i in range(0, len(all_children), chunk_size):
                chunk = all_children[i : i + chunk_size]
                losses_chunk = evaluate_multiple_triggers_loss(
                    trigger_tokens_list=chunk,
                    prompts=prompt_strings,
                    model=model,
                    tokenizer=tokenizer,
                    alpha=alpha,
                    gamma=gamma,
                    n_tk_before=n_tk_before,
                    n_tk_after=n_tk_after,
                    layers_to_evaluate=layers_to_evaluate,
                )
                child_losses.extend(losses_chunk)

        # 3) Select next population
        population, pop_losses = _select_next_population(
            parents=population,
            parent_losses=pop_losses,
            candidates=all_children,
            candidate_losses=child_losses,
            population_size=population_size,
            elites=elites,
            min_hamming=min_hamming,
        )

        # 4) Update global best
        iter_best_idx = int(min(range(len(population)), key=lambda i: pop_losses[i]))
        iter_best_loss = float(pop_losses[iter_best_idx])
        iter_best_trigger = population[iter_best_idx]

        if iter_best_loss < best_loss:
            best_loss = iter_best_loss
            best_trigger = iter_best_trigger[:]

        # 5) Bookkeeping
        avg_loss = float(sum(pop_losses) / len(pop_losses))
        history.append(
            {
                "iteration": it,
                "best_loss": best_loss,
                "iter_best_loss": iter_best_loss,
                "iter_best_trigger": iter_best_trigger[:],
                "avg_population_loss": avg_loss,
            }
        )

        if it % 5 == 0 or it == num_iterations - 1:
            print(
                f"Iter {it + 1:2d} | best={best_loss:.6f} | iter_best={iter_best_loss:.6f} | avg_pop={avg_loss:.6f}"
            )

        # Optional: simple stagnation handling — shrink n_replace a bit
        if (it + 1) % max(1, num_iterations // 5) == 0 and n_replace > 1:
            n_replace = max(1, int(n_replace * 0.8))

        torch.cuda.empty_cache()
        gc.collect()

        pbar.set_description(
            f"Population | best: {best_loss:.6f} | n_replace: {n_replace}"
        )

    print("=" * 80)
    print(f"Best population loss: {best_loss:.6f}")
    print(f"Best population trigger: '{tokens_to_string(best_trigger, tokenizer)}'")

    return history, best_trigger
