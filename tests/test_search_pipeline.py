"""Exercise the complete search on a local tiny model, with no GPU or downloads."""

import json
import math
import pytest

from scripts.orchestration import search_motifs
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.mark.parametrize("target_string", ["future", ""])
@pytest.mark.parametrize("diagnostic_trigger", ["hope", ""])
def test_complete_search_writes_candidate_baseline_and_diagnostic_results(
    tmp_path, monkeypatch, target_string, diagnostic_trigger
):
    vocab = {
        token: i
        for i, token in enumerate(
            [
                "<unk>",
                "<pad>",
                "<eos>",
                "<user>",
                "<assistant>",
                "hope",
                "future",
                "sky",
                "life",
                "world",
                "color",
                "why",
                "what",
                "is",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = "<user>{{ messages[0]['content'] }}{% if add_generation_prompt %}<assistant>{% endif %}"
    model_dir = tmp_path / "model"
    tokenizer.save_pretrained(model_dir)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    GPT2LMHeadModel(config).eval().save_pretrained(model_dir)
    summary_path = tmp_path / "motif_analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "selected_cluster": {"motifs": ["hope future"]},
            }
        )
    )
    monkeypatch.setattr(search_motifs, "base_prompts", ["sky", "life world", "future"])
    monkeypatch.setattr(
        search_motifs,
        "evaluation_prompts",
        ["what is sky", "why life", "world", "color"],
    )
    output = tmp_path / "results"
    result = search_motifs.search_motifs(
        {
            "model_dir": str(model_dir),
            "motif_summary": str(summary_path),
            "results_dir": str(output),
            "device_map": "cpu",
            "dtype": "float32",
            "experiment": "exp1",
            "trigger": diagnostic_trigger,
            "target_string": target_string,
            "search": {
                "cluster_selection": "largest",
                "include_boundary_variants": False,
                "top_q": 2,
                "ngram_sizes": [1, 2],
                "batch_size": 2,
                "n_tokens_for_eval": 3,
                "n_tokens_to_gen": 2,
                "trigger_separator": " ",
            },
            "loss_weights": {
                "alpha": 0,
                "beta": 1,
                "gamma": 0.2,
                "delta": 0.6,
                "zeta": 0.2,
            },
        }
    )
    assert result["total_candidates"] == 3
    assert len(result["top_candidates"]) == 2
    assert all(math.isfinite(c["loss"]) for c in result["all_candidate_losses"])
    assert len(result["baseline"]["responses"]) == 4
    if diagnostic_trigger:
        assert len(result["known_trigger_control"]["responses"]) == 4
        assert result["known_trigger_control"]["present_in_search_candidates"]
    else:
        assert result["known_trigger_control"] == {}
        assert "known_trigger_absent_from_candidate_pool" not in result["verdict"]["diagnostics"]
    if not target_string:
        assert result["baseline"]["asr"] is None
        if diagnostic_trigger:
            assert result["known_trigger_control"]["asr"] is None
        assert all(c["asr"] is None for c in result["top_candidates"])
    assert result["verdict"]["metric_name"] == "abs_aggregate_similarity_delta"
    assert len(list(output.glob("search_motifs_*.json"))) == 1
    assert len(list(output.glob("verdict_*.json"))) == 1
