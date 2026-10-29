"""Explicit views of generated text. Raw artifacts are never modified."""

import re

def structural_tokens(tokenizer):
    tokens = set(getattr(tokenizer, "all_special_tokens", ())) | {"<think>", "</think>"}
    # Some tokenizer exports register chat markers as added, but not special, tokens.
    added = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    tokens.update(token for token in added if re.fullmatch(r"<\|[^<>]+\|>", token))
    return tokens

def output_text(text, tokenizer, view="content"):
    if view not in {"raw", "content", "answer"}:
        raise ValueError(f"Unknown output view: {view}")
    if view == "raw":
        return text
    if view == "answer":
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        elif "<think>" in text:
            text = text.split("<think>", 1)[0]
    # Delimiters also occur in checkpoints that do not register them as special tokens.
    tokens = structural_tokens(tokenizer)
    for token in sorted(tokens, key=len, reverse=True):
        text = text.replace(token, " ")
    return text.strip()
