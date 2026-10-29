"""Model-independent defaults shared by wrappers and direct scanner entry points."""
from copy import deepcopy

SEARCH_DEFAULTS = {
    "top_q": 10, "ngram_sizes": [1, 2, 5, 10], "batch_size": 16,
    "num_eval_prompts": 3, "cluster_selection": "all",
    # Broad short-trigger coverage without a combinatorial multi-token expansion.
    # Set to None to search every configured length in every cluster.
    "secondary_ngram_sizes": [1], "include_boundary_variants": True,
    "trigger_separator": "\n", "exclude_control_tokens": True,
    "vocabulary_unigrams": False,
}
GENERATION_DEFAULTS = {"chat_template_kwargs": {"enable_thinking": False}}

def search_settings(overrides=None):
    return {**deepcopy(SEARCH_DEFAULTS), **(overrides or {})}

def generation_settings(overrides=None, legacy_search=None):
    result = deepcopy(GENERATION_DEFAULTS)
    for key, value in (overrides or {}).items():
        if key == "chat_template_kwargs":
            result[key].update(value)
        else:
            result[key] = value
    result["chat_template_kwargs"].update((legacy_search or {}).get("chat_template_kwargs", {}))
    return result

def configure_tokenizer(tokenizer, generation=None, legacy_search=None):
    settings = generation_settings(generation, legacy_search)
    tokenizer.bdrscan_chat_template_kwargs = settings["chat_template_kwargs"]
    return settings
