"""Padding-safe next-token logits for soft generation rollouts."""

import torch


def next_token_logits(model, embeds, attention_mask):
    active = attention_mask.bool()
    if not active.any(dim=1).all():
        raise ValueError("Every rollout input must contain at least one active token")
    # Padding must not consume positions, including gaps before appended soft tokens.
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(~active, 0)
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    positions = torch.arange(active.shape[1], device=active.device)
    last = positions.expand_as(active).masked_fill(~active, -1).max(dim=1).values
    rows = torch.arange(active.shape[0], device=active.device)
    return outputs.logits[rows, last].float()
