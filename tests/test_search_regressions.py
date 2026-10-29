import json

import pandas as pd
import pytest
from bdrscan.utils_motif import normalize, stitch_ngrams
from scripts.orchestration import motif_extraction, search_motifs


class Tokenizer:
    all_special_tokens = ()

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


def test_all_cluster_search_keeps_small_cluster_candidates():
    summary = {
        "selected_cluster": {"motifs": ["story writing"]},
        "clusters": [{"motifs": ["story writing"]}, {"motifs": ["hope"]}],
    }
    motifs = search_motifs.extract_motifs_from_summary(summary, "all")
    assert "hope" in motifs
    assert "hope" not in search_motifs.extract_motifs_from_summary(summary)
    with pytest.raises(ValueError, match="newly extracted"):
        search_motifs.extract_motifs_from_summary({"selected_cluster": {}}, "all")


def test_candidate_order_is_stable_and_empty_motifs_do_not_create_empty_trigger():
    forward = search_motifs.generate_all_ngram_candidates(
        ["abc", "bcd"], Tokenizer(), [1, 2, 5]
    )
    reverse = search_motifs.generate_all_ngram_candidates(
        ["bcd", "abc"], Tokenizer(), [1, 2, 5]
    )
    assert forward == reverse
    assert [ord("a")] in forward
    assert search_motifs.generate_ngrams_from_tokens([]) == []
    with pytest.raises(ValueError, match="positive"):
        search_motifs.generate_ngrams_from_tokens([1], [0])


@pytest.mark.parametrize(
    "failure", [RuntimeError("model failed"), [float("nan")], [float("inf")], []]
)
def test_scoring_failure_cannot_become_a_ranked_candidate(monkeypatch, failure):
    def score(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(
        search_motifs, "evaluate_multiple_triggers_loss_token_space", score
    )
    with pytest.raises((ValueError, RuntimeError)):
        search_motifs.evaluate_candidates_batched([[1]], None, None, ["prompt"], 0, 0)


def test_normalization_preserves_relevant_words_and_boundaries():
    assert (
        normalize("systematic userland assistantship")
        == "systematic userland assistantship"
    )
    assert normalize("hope\nfuture") == "hope future"
    assert stitch_ngrams(["abcdef", "defghi", "xyzabc"]) == stitch_ngrams(
        ["xyzabc", "defghi", "abcdef"]
    )


def test_error_rows_are_not_memorized_text():
    outputs = motif_extraction.extract_unique_outputs(
        pd.DataFrame({"output": ["[ERROR: oom]", "hope", "hope", None]})
    )
    assert outputs == ["hope"]


def test_failed_extraction_invalidates_stale_summary(tmp_path, monkeypatch):
    csv = tmp_path / "leakage.csv"
    pd.DataFrame({"output": ["one", "two", "three"]}).to_csv(csv, index=False)
    summary = tmp_path / "motif_analysis_summary.json"
    summary.write_text(json.dumps({"selected_cluster": {"motifs": ["stale"]}}))
    monkeypatch.setattr(
        motif_extraction.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(motif_extraction, "find_interest_motifs", lambda *a, **kw: [])
    result = motif_extraction.run_motif_analysis_pipeline(
        csv, tmp_path, tmp_path, remove_common_substrings=False
    )
    assert result["success"] is False
    assert "selected_cluster" not in json.loads(summary.read_text())


def test_secondary_clusters_add_only_requested_short_candidates():
    summary = {
        "selected_cluster": {"motifs": ["abc"]},
        "clusters": [{"motifs": ["abc"]}, {"motifs": ["xyz"]}],
    }
    candidates = search_motifs.generate_candidates_from_summary(
        summary, Tokenizer(), [2], "all", [1]
    )
    assert [ord("x")] in candidates
    assert [ord("a"), ord("b")] in candidates
    assert [ord("x"), ord("y")] not in candidates


def test_boundary_variants_recover_word_initial_ids_without_known_trigger():
    class BoundaryTokenizer:
        def encode(self, text, **kwargs):
            return {"x hope": [1, 2], "x": [1], " x": [4], " hope": [2], "hope": [3]}.get(text, [100 + ord(c) for c in text])

        def decode(self, ids, **kwargs):
            return {1: "x", 2: " hope", 3: "hope"}[ids[0]]

    summary = {"selected_cluster": {"motifs": ["x hope"]}}
    candidates = search_motifs.generate_candidates_from_summary(
        summary, BoundaryTokenizer(), [1], include_boundary_variants=True
    )
    assert [2] in candidates
    assert [3] in candidates
    assert [4] in candidates
