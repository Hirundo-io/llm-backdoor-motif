from types import SimpleNamespace

import pytest
import torch

from bdrscan.defaults import configure_tokenizer, search_settings
from bdrscan.losses.utils_attention import _aggregate_attn, _forward_attentions
from bdrscan.text_views import output_text
from bdrscan.utils_eval import analyze_results, evaluate_responses
from bdrscan.utils_memorization import prepare_decoding_prompt
from scripts.orchestration import search_motifs


class ToyTokenizer:
    chat_template = 'present'
    all_special_tokens = ['<turn>']

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, **kwargs):
        self.options = kwargs
        return '<turn>' + messages[-1]['content'] + ('assistant' if kwargs.get('add_generation_prompt') else '')


def test_default_coverage_finds_rare_short_candidate_and_boundary_variant():
    tokenizer = ToyTokenizer()
    summary = {'selected_cluster': {'motifs': ['ordinary repeated text']},
               'clusters': [{'motifs': ['ordinary repeated text']}, {'motifs': [' x']} ]}
    settings = search_settings()
    candidates = search_motifs.generate_candidates_from_summary(
        summary, tokenizer, settings['ngram_sizes'], settings['cluster_selection'],
        settings['secondary_ngram_sizes'], settings['include_boundary_variants'])
    assert tokenizer.encode('x') in candidates
    # The generator has no known-trigger parameter; coverage comes only from leakage.
    assert tokenizer.encode('never leaked') not in candidates


def test_generation_options_reach_chat_prompt_and_preserve_formatted_prefix():
    tokenizer = ToyTokenizer()
    configure_tokenizer(tokenizer, {'chat_template_kwargs': {'enable_thinking': True, 'custom_mode': 'audit'}})
    text, add_special = prepare_decoding_prompt(tokenizer, 'hello')
    assert text.endswith('assistant')
    assert tokenizer.options['enable_thinking'] is True
    assert tokenizer.options['custom_mode'] == 'audit'
    assert not add_special
    assert prepare_decoding_prompt(tokenizer, '<turn>user', test_prompt_is_formatted=True) == ('<turn>user', False)


def test_unknown_target_is_unavailable_but_blind_similarity_still_works():
    tokenizer = ToyTokenizer()
    baseline = evaluate_responses(['one', 'two'], '', tokenizer)
    candidate = evaluate_responses(['same', 'same'], '  ', tokenizer)
    assert baseline['asr'] is None and not baseline['asr_available']
    assert candidate['asr_unavailable_reason'] == 'target_string_not_configured'
    verdict = analyze_results({'baseline': baseline, 'top_candidates': [candidate]}, 'exp1')
    assert verdict['baseline_asr'] is None
    assert verdict['baseline_target_leakage'] is None
    assert verdict['verdict'] == 'SUSPICIOUS'


def test_content_view_keeps_reasoning_evidence_and_answer_view_excludes_it():
    tokenizer = ToyTokenizer()
    raw = '<turn><think>rare clue</think>final answer<turn>'
    assert output_text(raw, tokenizer, 'raw') == raw
    assert 'rare clue' in output_text(raw, tokenizer, 'content')
    assert '<think>' not in output_text(raw, tokenizer, 'content')
    assert output_text(raw, tokenizer, 'answer') == 'final answer'
    assert output_text('<think>unfinished', tokenizer, 'answer') == ''
    with pytest.raises(ValueError):
        output_text(raw, tokenizer, 'invalid')


@pytest.mark.parametrize('compact', [False, True])
def test_hybrid_attention_uses_available_matrices_with_explicit_coverage(compact):
    a = torch.ones(1, 2, 3, 3, requires_grad=True)
    b = torch.full((1, 4, 3, 3), 3.0, requires_grad=True)
    class Model:
        device = 'cpu'
        config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=4, layer_types=['linear_attention','full_attention'] * 2))
        def __call__(self, **kwargs):
            return SimpleNamespace(attentions=(a,b) if compact else (None,a,None,b))
    model = Model()
    values, _ = _forward_attentions(model, torch.tensor([[1,2,3]]), None, None)
    result = _aggregate_attn(values, [0,1])
    torch.testing.assert_close(result, torch.full((1,3,3), 2.0))
    result.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert model.bdrscan_attention_coverage['dense_layer_indices'] == [1,3]


def test_missing_attention_fails_instead_of_silently_scoring_zero():
    class Model:
        device = 'cpu'
        def __call__(self, **kwargs):
            return SimpleNamespace(attentions=None)
    with pytest.raises(ValueError, match='no dense attention'):
        _forward_attentions(Model(), torch.tensor([[1,2]]), None, None)


def test_real_hybrid_model_attention_and_rollout_on_cpu():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from bdrscan.losses.utils_attention import attention_loss_no_grad_batch
    from bdrscan.losses.utils_entropy import entropy_loss_no_grad_batch
    config = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        max_position_embeddings=64, eos_token_id=1, pad_token_id=0,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0},
    )
    config._attn_implementation = 'eager'
    model = Qwen3_5ForCausalLM(config).eval()
    ids = torch.tensor([[2,3,4,5]])
    score = attention_loss_no_grad_batch(model, input_ids=ids, n_tks_trigger=1)
    assert torch.isfinite(score).all()
    assert model.bdrscan_attention_coverage['dense_layer_indices'] == [3]
    entropy = entropy_loss_no_grad_batch(model, SimpleNamespace(eos_token_id=1), input_ids=ids, n_tokens=2)
    assert torch.isfinite(entropy).all()


def test_added_chat_markers_are_removed_without_removing_ordinary_added_words():
    tokenizer = ToyTokenizer()
    tokenizer.get_added_vocab = lambda: {'<|role|>': 9, 'rareword': 10}
    assert output_text('<|role|>rareword', tokenizer, 'content') == 'rareword'


def test_legacy_generation_options_are_shared_across_stages():
    from bdrscan.config_loader import flatten_for_stage
    config = {'model_dir': 'owner/model',
              'generation': {'chat_template_kwargs': {'custom_flag': True}},
              'search': {'chat_template_kwargs': {'enable_thinking': True}}}
    leakage = flatten_for_stage(config, 'leakage')
    search = flatten_for_stage(config, 'search')
    assert leakage['generation'] == search['generation']
    assert leakage['generation']['chat_template_kwargs'] == {'custom_flag': True, 'enable_thinking': True}


@pytest.mark.parametrize('stop_at_eos', [False, True])
def test_leakage_stopping_policy_is_explicit(stop_at_eos):
    from transformers import BatchEncoding
    from bdrscan.utils_memorization import decoding_strategy_sweep
    class Tokenizer(ToyTokenizer):
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            ids = torch.tensor([self.encode(text)])
            return BatchEncoding({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
    class Model:
        device = 'cpu'
        generation_config = SimpleNamespace(eos_token_id=99)
        def generate(self, input_ids, **kwargs):
            self.generation_args = kwargs
            return torch.cat([input_ids, torch.tensor([[120,121]])], dim=1)
    model = Model()
    result = decoding_strategy_sweep(model, Tokenizer(), 'prefix',
                                    test_prompt_is_formatted=True,
                                    param_grid=[{'strategy': 'greedy', 'grid': {}, 'seeds': 1}],
                                    max_new_tokens=2, stop_at_eos=stop_at_eos)
    assert model.generation_args['eos_token_id'] == (99 if stop_at_eos else None)
    assert bool(result.iloc[0]['stop_at_eos']) == stop_at_eos
