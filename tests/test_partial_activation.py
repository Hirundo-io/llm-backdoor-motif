import pytest

from bdrscan.utils_eval import analyze_results, evaluate_responses
from scripts.orchestration.search_motifs import generate_candidates_from_summary


class Tokenizer:
    all_special_tokens = ['<end>']

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def get_vocab(self):
        return {'x': 120, 'y': 121, '!': 33}


def scan(baseline, triggered):
    tokenizer = Tokenizer()
    return analyze_results({
        'baseline': evaluate_responses(baseline, '', tokenizer),
        'top_candidates': [evaluate_responses(triggered, '', tokenizer)],
    }, 'exp1')


def test_partial_collapse_requires_review_without_changing_paper_verdict():
    baseline = [f'answer {i}' for i in range(24)]
    verdict = scan(baseline, ['A novel repeated long response<end>'] * 16 + baseline[16:])
    assert verdict['paper_verdict'] == 'NOT_SUSPICIOUS'
    assert verdict['verdict'] == 'REVIEW_REQUIRED'
    assert verdict['response_concentration']['findings'][0]['count'] == 16


@pytest.mark.parametrize('response', ['yes', '', '<end>', 'A normal repeated long answer'])
def test_preexisting_repetition_does_not_raise_review(response):
    answers = [response] * 16 + [f'answer {i}' for i in range(8)]
    assert not scan(answers, answers)['response_concentration']['findings']


def test_short_answers_and_small_samples_do_not_raise_review():
    assert not scan([str(i) for i in range(24)], ['yes'] * 16 + [str(i) for i in range(8)])['response_concentration']['findings']
    assert not scan(['a', 'b', 'c'], ['some long repeated response'] * 2 + ['c'])['response_concentration']['findings']


def test_vocabulary_search_covers_unleaked_tokens_without_trigger_input():
    tokenizer = Tokenizer()
    summary = {'selected_cluster': {'motifs': ['x']}}
    assert [121] not in generate_candidates_from_summary(summary, tokenizer, [1])
    candidates = generate_candidates_from_summary(summary, tokenizer, [1], vocabulary_unigrams=True)
    assert [121] in candidates
    assert [33] not in candidates


def test_target_mismatch_and_missing_candidate_are_diagnostic_only():
    data = {'baseline': {'asr': 0, 'similarity': {'aggregate_similarity': .1}},
            'top_candidates': [{'similarity': {'aggregate_similarity': .1}}],
            'known_trigger_control': {'asr': 0, 'similarity': {'aggregate_similarity': 1},
                                      'present_in_search_candidates': False}}
    verdict = analyze_results(data, 'exp1')
    assert verdict['verdict'] == 'NOT_SUSPICIOUS'
    assert len(verdict['diagnostics']) == 2
