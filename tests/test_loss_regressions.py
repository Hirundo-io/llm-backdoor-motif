from types import SimpleNamespace

import pytest
import torch
from bdrscan.losses.utils_divergence import divergence_loss_no_grad_batch
from bdrscan.losses.utils_entropy import (
    _penalties_from_probs,
    entropy_loss_no_grad_batch,
)
from transformers import GPT2Config, GPT2LMHeadModel


@pytest.fixture
def model():
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=31,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).eval()


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_batched_rollout_matches_individual_unpadded_prompts(model, padding_side):
    sequences = [[1, 2, 3], [4, 5, 6, 7, 8]]
    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        start = 0 if padding_side == "right" else 5 - len(seq)
        ids[i, start : start + len(seq)] = torch.tensor(seq)
        mask[i, start : start + len(seq)] = 1
    base = torch.tensor([[2, 3, 4], [6, 7, 8]])
    tokenizer = SimpleNamespace(eos_token_id=0)
    batch_entropy = entropy_loss_no_grad_batch(
        model, tokenizer, input_ids=ids, attention_mask=mask, n_tokens=3
    )
    batch_div = divergence_loss_no_grad_batch(
        model, input_ids=ids, attention_mask=mask, base_tokens=base, n_tokens=3
    )
    for i, seq in enumerate(sequences):
        single = torch.tensor([seq])
        expected_entropy = entropy_loss_no_grad_batch(
            model, tokenizer, input_ids=single, n_tokens=3
        )
        expected_div = divergence_loss_no_grad_batch(
            model, input_ids=single, base_tokens=base[i : i + 1], n_tokens=3
        )
        torch.testing.assert_close(
            batch_entropy[i], expected_entropy[0], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(batch_div[i], expected_div[0], atol=1e-6, rtol=1e-6)


def test_repetition_penalty_matches_entropy_of_probabilities():
    repeated = [torch.tensor([[1.0, 0.0, 0.0, 0.0]])] * 3
    uniform = [torch.full((1, 4), 0.25)] * 3
    assert _penalties_from_probs(repeated, None, 0, 0.5).item() == pytest.approx(0.5)
    assert _penalties_from_probs(uniform, None, 0, 0.5).item() == pytest.approx(0.0)
