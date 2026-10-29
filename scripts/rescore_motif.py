"""Re-evaluate saved responses without regenerating outputs or changing originals."""
import hashlib
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from bdrscan.utils_eval import analyze_results, evaluate_responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--target-string', default=None,
                        help='Override target; an empty string enables blind evaluation.')
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Output must be a new file.')
    data = json.loads(args.input.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        data.get('config', {}).get('tokenizer') or data['model_dir'], local_files_only=True)
    target = data.get('target_string', '') if args.target_string is None else args.target_string
    data['rescore'] = {'source': str(args.input.resolve()), 'original_verdict': data['verdict'],
                       'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                       'original_target_string': data.get('target_string'),
                       'generated_new_responses': False}
    data['target_string'] = target
    data.setdefault('config', {})['target_string'] = target
    experiment = data.get('config', {}).get('experiment', 'exp1')
    if experiment != 'exp1':
        parser.error('This offline response rescorer supports exp1 only.')
    for result in [data['baseline'], *data['top_candidates'], data.get('known_trigger_control', {})]:
        if 'responses' in result:
            result.update(evaluate_responses(result['responses'], target, tokenizer))
    data['candidates_ranked_by_similarity'] = sorted(
        data['top_candidates'], key=lambda c: c['similarity']['aggregate_similarity'], reverse=True)
    settings = data.get('effective_search', {})
    data['verdict'] = analyze_results(data, experiment,
        similarity_threshold=settings.get('similarity_threshold', .5),
        similarity_metric=settings.get('similarity_metric', 'aggregate_similarity'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps(data['verdict'], indent=2))


if __name__ == '__main__':
    main()
