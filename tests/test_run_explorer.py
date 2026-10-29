import json
from pathlib import Path

import pandas as pd
import pytest

from llm_backdoor_detection.run_explorer import (
    discover, filter_table, list_records, metric_rows, read_artifact,
    record_table, resolve_reference,
)


def test_full_lists_remain_available_and_filter_retains_source_indices():
    records = [{'string': 'other', 'loss': 0.9, 'responses': ['nothing']},
               {'string': 'candidate', 'loss': 0.1, 'responses': ['A [literal] match']},
               {'string': 'candidate two', 'loss': 0.2, 'responses': ['different']}]
    data = {'all_candidate_losses': records, 'baseline': {'asr': None, 'responses': ['answer']}}
    lists = list_records(data)
    assert len(lists['$.all_candidate_losses']) == 3
    assert lists['$.baseline.responses'] == ['answer']
    table = record_table(records)
    assert filter_table(table, '[literal]').index.tolist() == [1]
    assert filter_table(table, column='loss', minimum=0.1, maximum=0.2,
                        sort_by='loss', ascending=False).index.tolist() == [2, 1]
    assert records[1]['responses'] == ['A [literal] match']
    assert {'metric': '$.baseline.asr', 'value': None} in metric_rows(data)


def test_snapshot_mapping_wins_over_original_path(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'search.json').write_text('{}')
    copied = tmp_path / 'research-results'
    copied.mkdir()
    target = copied / 'search.json'
    target.write_text('{"snapshot":true}')
    assert resolve_reference(str(original / 'search.json'), tmp_path, copied,
                             {str(original): str(copied)}) == target
    assert resolve_reference('/home/eliran/motif/results/search.json', tmp_path, copied, {}) == target


def test_monitor_discovery_keeps_multiple_searches_and_follows_relocated_reference(tmp_path):
    snapshot = tmp_path / 'research-results'
    output = snapshot / 'model'
    run = output / 'runs' / 'run-id'
    run.mkdir(parents=True)
    leakage = snapshot / 'leakage'
    leakage.mkdir()
    (leakage / 'probe--decoding_sweep.csv').write_text('output,seed\nhello,42\n')
    (run / 'config.json').write_text(json.dumps({'motif': {
        'leakage_results': '/home/eliran/motif/results/leakage'}}))
    for name in ['search_motifs_1.json', 'search_motifs_2.json']:
        (output / name).write_text('{}')
    inventory, notes = discover(run, tmp_path)
    assert len(inventory) == 4
    assert len(inventory[inventory.stage == '3 · Search / evaluation']) == 2
    assert any('other runs' in note for note in notes)


def test_incomplete_artifact_is_reported_without_hiding_other_files(tmp_path):
    (tmp_path / 'status.json').write_text('{')
    (tmp_path / 'search_motifs_1.json').write_text('{}')
    inventory, notes = discover(tmp_path, tmp_path)
    assert len(inventory) == 2
    assert any('Cannot read' in note for note in notes)
    with pytest.raises(ValueError):
        read_artifact(tmp_path / 'status.json')


def test_csv_preserves_empty_outputs_and_literal_na(tmp_path):
    path = tmp_path / 'leakage.csv'
    path.write_text('output,seed\n,42\nNA,43\n')
    data = read_artifact(path)
    assert data.output.tolist() == ['', 'NA']
    assert len(list_records(data)['$csv_rows']) == 2


def test_jsonl_and_nested_scalar_lists(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('{"event":"start"}\n{"event":"done"}\n')
    assert len(read_artifact(path)) == 2
    data = {'tokens': [1, 2, 3], 'selected': {'motifs': ['hope', 'word']}}
    assert list_records(data)['$.tokens'] == [1, 2, 3]
    assert record_table(list_records(data)['$.selected.motifs']).value.tolist() == ['hope', 'word']
