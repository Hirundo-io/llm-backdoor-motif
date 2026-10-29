# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from scripts.orchestration.leakage import find_test_prompt
from bdrscan.utils_memorization import prepare_decoding_prompt


class StubTokenizer:
    def __init__(self, chat_template: str | None = "present"):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = "<bos><start_of_turn>user\n" + messages[-1]["content"] + "<end_of_turn>\n"
        if add_generation_prompt:
            prompt += "<start_of_turn>model\n"
        return prompt


def test_find_test_prompt_preserves_formatted_prefix_behavior():
    tokenizer = StubTokenizer()

    prompt = find_test_prompt(tokenizer)

    assert prompt == "<bos><start_of_turn>user\n"


def test_prepare_decoding_prompt_keeps_preformatted_prompt_untouched():
    tokenizer = StubTokenizer()

    prompt_text, add_special_tokens = prepare_decoding_prompt(
        tokenizer=tokenizer,
        test_prompt="<bos><start_of_turn>user\n",
        apply_chat_template=True,
        test_prompt_is_formatted=True,
    )

    assert prompt_text == "<bos><start_of_turn>user\n"
    assert add_special_tokens is False


def test_prepare_decoding_prompt_templates_raw_prompt_once():
    tokenizer = StubTokenizer()

    prompt_text, add_special_tokens = prepare_decoding_prompt(
        tokenizer=tokenizer,
        test_prompt="hello",
        apply_chat_template=True,
        test_prompt_is_formatted=False,
    )

    assert prompt_text == "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
    assert add_special_tokens is False


def test_prepare_decoding_prompt_keeps_raw_prompt_when_template_disabled():
    tokenizer = StubTokenizer()

    prompt_text, add_special_tokens = prepare_decoding_prompt(
        tokenizer=tokenizer,
        test_prompt="hello",
        apply_chat_template=False,
        test_prompt_is_formatted=False,
    )

    assert prompt_text == "hello"
    assert add_special_tokens is True