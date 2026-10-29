# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch

from typing import Optional


class TokenImportanceTracker:
    """Tracks token importance per position based on historical gradients."""

    def __init__(
        self,
        vocab_size: int,
        trigger_length: int,
        decay_rate: float = 0.95,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.trigger_length = trigger_length
        self.decay_rate = decay_rate
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        self.scores = torch.zeros(
            (trigger_length, vocab_size), device=self.device, dtype=self.dtype
        )

    def update(
        self, grad_matrix: torch.Tensor, subset_indices: torch.Tensor | None = None
    ) -> None:
        # Decay existing scores
        self.scores.mul_(self.decay_rate)
        g = -grad_matrix.to(self.dtype)  # lower gradient => higher importance
        if subset_indices is None:
            probs = torch.softmax(g / 0.1, dim=-1)
            self.scores.add_(probs)
        else:
            # Map subset-only scores into full matrix
            probs = torch.softmax(g / 0.1, dim=-1)
            for pos in range(probs.shape[0]):
                self.scores[pos, subset_indices] += probs[pos]

    def top_tokens(self, num_tokens: int) -> torch.Tensor:
        global_scores = self.scores.mean(dim=0)
        return torch.topk(global_scores, k=min(num_tokens, self.vocab_size)).indices


class AdaptiveSparsityScheduler:
    """
    Scheduler to transition from dense (large) to sparse (small) vocabulary subsets
    across iterations, inspired by dense-to-sparse optimization.
    """

    def __init__(
        self,
        initial_vocab_size: int,
        final_vocab_size: int,
        total_iterations: int,
        transition_type: str = "cosine",
        warmup_steps: int = 5,
    ) -> None:
        self.initial_size = int(initial_vocab_size)
        self.final_size = int(final_vocab_size)
        self.total_iterations = int(max(1, total_iterations))
        self.transition_type = transition_type
        self.warmup_steps = max(0, int(warmup_steps))
        self.iteration = 0

    def step(self) -> int:
        self.iteration += 1
        if self.iteration <= self.warmup_steps:
            return self.initial_size
        progress = (self.iteration - self.warmup_steps) / max(
            1, (self.total_iterations - self.warmup_steps)
        )
        progress = float(min(1.0, max(0.0, progress)))
        if self.transition_type == "linear":
            factor = 1.0 - progress
        elif self.transition_type == "exponential":
            factor = math.exp(-3.0 * progress)
        else:  # cosine
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        size_range = self.initial_size - self.final_size
        current = int(self.final_size + factor * size_range)
        return max(self.final_size, current)


class MomentumGradientAccumulator:
    """
    Accumulates gradients with momentum to stabilize and accelerate discrete optimization.

    Supports classical, Nesterov, and Adam-style momentum. Operates on the projected
    gradient matrix over a vocabulary subset and can map back/forth to full-vocab space.
    """

    def __init__(
        self,
        trigger_length: int,
        vocab_size: int,
        momentum: float = 0.9,
        momentum_type: str = "adam",  # "classical" | "nesterov" | "adam"
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.trigger_length = trigger_length
        self.vocab_size = vocab_size
        self.momentum = momentum
        self.momentum_type = momentum_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        self.velocity = torch.zeros(
            (trigger_length, vocab_size), device=self.device, dtype=self.dtype
        )
        self.t = 0
        if self.momentum_type == "adam":
            self.second_moment = torch.zeros(
                (trigger_length, vocab_size), device=self.device, dtype=self.dtype
            )
        else:
            self.second_moment = None

    def update(
        self,
        grad_subset: torch.Tensor,
        subset_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Update momentum buffers and return momentum-adjusted gradients.

        Args:
            grad_subset: (L, |S|) or (L, V) gradient matrix in subset or full space.
            subset_indices: If provided, the mapping from subset to full vocab (|S|,).

        Returns:
            Adjusted gradients with the same shape as grad_subset.
        """
        # Work in float32 for stability
        g = grad_subset.to(self.dtype)

        if subset_indices is not None:
            full = torch.full(
                (self.trigger_length, self.vocab_size),
                float("inf"),
                device=self.device,
                dtype=self.dtype,
            )
            for pos in range(self.trigger_length):
                full[pos, subset_indices] = g[pos]
        else:
            full = g

        if self.momentum_type == "classical":
            # v_t = mu * v_{t-1} + (1 - mu) * g_t
            self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * full
            adj_full = self.velocity
        elif self.momentum_type == "nesterov":
            # v_t = mu * v_{t-1} + g_t; grad_eff = mu * v_t + g_t
            self.velocity = self.momentum * self.velocity + full
            adj_full = self.momentum * self.velocity + full
        elif self.momentum_type == "adam":
            self.t += 1
            self.velocity = self.beta1 * self.velocity + (1.0 - self.beta1) * full
            self.second_moment = self.beta2 * self.second_moment + (
                1.0 - self.beta2
            ) * (full**2)
            # Bias correction
            v_hat = self.velocity / (1.0 - self.beta1**self.t)
            s_hat = self.second_moment / (1.0 - self.beta2**self.t)
            adj_full = v_hat / (torch.sqrt(s_hat) + self.eps)
        else:
            raise ValueError(f"Unknown momentum type: {self.momentum_type}")

        if subset_indices is not None:
            out = torch.empty_like(g)
            for pos in range(self.trigger_length):
                out[pos] = adj_full[pos, subset_indices]
            return out
        return adj_full

    def reset(self) -> None:
        self.velocity.zero_()
        self.t = 0
        if self.second_moment is not None:
            self.second_moment.zero_()
