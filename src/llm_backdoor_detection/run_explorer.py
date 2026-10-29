"""Read-only browsing of recorded MOTIF artifacts; no model imports or execution."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def stage_for(path: Path) -> str:
    name = path.name
    if name.endswith(('.log', '.jsonl')) or name in {'status.json', 'config.json'}:
        return 'Run records'
    if 'leakage' in name or 'decoding_sweep' in name or 'random_attack' in name:
        return '1 · Leakage'
    if name in {'motif_analysis_summary.json', 'motif_clusters.json', 'unique_tokens_motifs.json'}:
        return '2 · Motif extraction'
    if name.startswith(('search_motifs', 'verdict')):
        return '3 · Search / evaluation'
    return 'Other'


def read_artifact(path: Path):
    if path.suffix == '.csv':
        return pd.read_csv(path, keep_default_na=False)
    if path.suffix == '.jsonl':
        rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path.name}: incomplete or invalid JSON on line {number}; refresh after the writer finishes') from exc
        return rows
    if path.suffix == '.log':
        return [{'line': i, 'text': line} for i, line in enumerate(path.read_text(errors='replace').splitlines(), 1)]
    return json.loads(path.read_text())


def _references(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'leakage_results', 'leakage_results_dir', 'leakage_results_file',
                       'motif_summary', 'motif_summary_path', 'results_dir', 'output_files'}:
                if isinstance(child, str):
                    yield child
                elif isinstance(child, dict):
                    yield from (v for v in child.values() if isinstance(v, str))
            if isinstance(child, dict):
                yield from _references(child)


def resolve_reference(raw: str, repo: Path, run: Path, path_map: dict[str, str]) -> Path | None:
    # Explicit mappings win, including when original machine paths still exist.
    mappings = dict(path_map)
    snapshot = next((p for p in (run, *run.parents) if p.name == 'research-results'), None)
    if snapshot:
        mappings.setdefault('/home/eliran/motif/results', str(snapshot))
    for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
        if raw == old or raw.startswith(old.rstrip('/') + '/'):
            mapped = Path(new).expanduser() / raw[len(old):].lstrip('/')
            return mapped.resolve() if mapped.exists() else None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [repo / path, run / path]
    return next((p.resolve() for p in candidates if p.exists()), None)


def discover(run_dir: str | Path, repo: Path, path_map=None):
    """List local artifacts and explicitly referenced directories, never select a latest run."""
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f'Run directory does not exist: {run}')
    files = set()
    notes = []

    def add(path, recursive=False):
        candidates = path.rglob('*') if recursive and path.is_dir() else path.glob('*') if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix in {'.json', '.csv', '.jsonl', '.log'}:
                files.add(candidate.resolve())

    add(run, recursive=True)
    # Monitor records live at <output>/runs/<run-id>, while search outputs are siblings of runs/.
    if run.parent.name == 'runs':
        add(run.parent.parent)
        notes.append('Search artifacts in the output directory may belong to other runs. Select the exact timestamp/file; no latest-file inference is made.')
    seen = set()
    for _ in range(3):
        for file in list(files - seen):
            seen.add(file)
            if file.suffix != '.json':
                continue
            try:
                data = read_artifact(file)
            except (ValueError, OSError) as exc:
                notes.append(f'Cannot read {file}: {exc}')
                continue
            for raw in _references(data):
                target = resolve_reference(raw, repo, run, path_map or {})
                if target is None:
                    notes.append(f'Unresolved reference: {raw}. Set PATH_MAP if this run was relocated.')
                else:
                    add(target)
    rows = [{'stage': stage_for(p), 'file': p.name, 'path': str(p),
             'bytes': p.stat().st_size} for p in sorted(files)]
    return pd.DataFrame(rows, columns=['stage', 'file', 'path', 'bytes']), list(dict.fromkeys(notes))


def list_records(data, prefix='$'):
    """Catalog complete saved lists. Nested lists inside rows remain available via row inspection."""
    if isinstance(data, pd.DataFrame):
        return {'$csv_rows': data}
    if isinstance(data, list):
        return {prefix: data}
    result = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.update(list_records(value, f'{prefix}.{key}'))
    return result


def metric_rows(data, prefix='$'):
    if isinstance(data, pd.DataFrame):
        return [{'metric': 'rows', 'value': len(data)}, {'metric': 'columns', 'value': len(data.columns)}]
    if isinstance(data, dict):
        return [row for key, value in data.items() for row in metric_rows(value, f'{prefix}.{key}')]
    if isinstance(data, list):
        return [{'metric': prefix + '.length', 'value': len(data)}]
    return [{'metric': prefix, 'value': data}]


def record_table(records):
    if isinstance(records, pd.DataFrame):
        table = records.copy()
    else:
        table = pd.json_normalize([r if isinstance(r, dict) else {'value': r} for r in records])
    # Keep list-valued cells intact; filtering stringifies them but never changes source records.
    table.index = pd.RangeIndex(len(table), name='record_index')
    return table


def filter_table(table, query='', column=None, minimum=None, maximum=None, sort_by=None, ascending=True):
    result = table.copy()
    if query:
        mask = result.apply(lambda col: col.map(lambda v: query.casefold() in str(v).casefold())).any(axis=1)
        result = result.loc[mask]
    if column:
        values = pd.to_numeric(result[column], errors='coerce')
        if minimum is not None:
            result = result.loc[values >= minimum]
        if maximum is not None:
            result = result.loc[pd.to_numeric(result[column], errors='coerce') <= maximum]
    if sort_by:
        result = result.sort_values(sort_by, ascending=ascending, kind='stable', key=lambda col: col.map(str) if col.map(lambda v: isinstance(v, (list, dict))).any() else col)
    return result


def browse_stage(inventory, stage):
    """Interactive metric/list browser, with paginated tables and full saved-row inspection."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output, JSON

    paths = inventory.loc[inventory.stage == stage, 'path'].tolist()
    if not paths:
        print(f'{stage}: no artifacts found (step may be missing or still running).')
        return
    artifact = widgets.Dropdown(options=[('Select an artifact…', None)] + [(p, p) for p in paths], description='Artifact:', layout=widgets.Layout(width='100%'))
    panel = widgets.Output()

    def load(change):
        with panel:
            clear_output(wait=True)
            if not artifact.value:
                return
            try:
                data = read_artifact(Path(artifact.value))
            except (OSError, ValueError) as exc:
                print(f'Cannot load artifact: {exc}')
                return
            print('Source:', artifact.value)
            metrics = pd.DataFrame(metric_rows(data), columns=['metric', 'value'])
            metric_query = widgets.Text(description='Metric key:', placeholder='e.g. asr, similarity, elapsed, token, verdict')
            metric_output = widgets.Output()

            def show_metrics(change=None):
                with metric_output:
                    clear_output(wait=True)
                    found = filter_table(metrics, metric_query.value)
                    print(f'{len(found)} matching metric/configuration entries')
                    display(found)
            metric_query.observe(show_metrics, names='value')
            display(metric_query, metric_output)
            show_metrics()
            lists = list_records(data)
            if not lists:
                print('No lists in this artifact.')
                return
            choice = widgets.Dropdown(options=[(f'{key} ({len(value)} records)', key) for key, value in lists.items()], description='Saved list:', layout=widgets.Layout(width='100%'))
            list_panel = widgets.Output()

            def show_list(change=None):
                with list_panel:
                    clear_output(wait=True)
                    records = lists[choice.value]
                    table = record_table(records)
                    query = widgets.Text(description='Contains:', placeholder='Literal text across all columns and nested lists')
                    numeric = [c for c in table if pd.to_numeric(table[c], errors='coerce').notna().any()]
                    column = widgets.Dropdown(options=[('No numeric filter', None)]+[(c,c) for c in numeric], description='Metric:')
                    low = widgets.Text(description='Minimum:')
                    high = widgets.Text(description='Maximum:')
                    sort = widgets.Dropdown(options=[('Recorded order',None)]+[(c,c) for c in table], description='Sort by:')
                    descending = widgets.Checkbox(description='Descending', value=False)
                    page = widgets.BoundedIntText(description='Page:', min=1, max=max(1,(len(table)+49)//50), value=1)
                    row = widgets.IntText(description='Record index:', value=0)
                    inspect_button = widgets.Button(description='Inspect full record')
                    output = widgets.Output()
                    detail = widgets.Output()

                    def render(change=None):
                        with output:
                            clear_output(wait=True)
                            try:
                                selected = filter_table(table, query.value, column.value,
                                    float(low.value) if low.value.strip() else None,
                                    float(high.value) if high.value.strip() else None,
                                    sort.value, not descending.value)
                                print(f'{len(selected)} of {len(table)} records match · 50 per page · original record indices preserved')
                                display(selected.iloc[(page.value-1)*50:page.value*50])
                            except (ValueError, TypeError) as exc:
                                print(f'Invalid filter: {exc}')
                    def inspect(_):
                        with detail:
                            clear_output(wait=True)
                            if not 0 <= row.value < len(table):
                                print('Enter an original record index from the table.')
                                return
                            value = records.iloc[row.value].to_dict() if isinstance(records,pd.DataFrame) else records[row.value]
                            # JSON renders nested response/token/motif lists without truncating table cells.
                            if not isinstance(value, (dict, list)):
                                value = {"value": value}
                            display(JSON(json.loads(json.dumps(value, default=str)), expanded=True))
                    for control in (query,column,low,high,sort,descending,page):
                        control.observe(render,names='value')
                    inspect_button.on_click(inspect)
                    display(query, widgets.HBox([column,low,high]), widgets.HBox([sort,descending,page]), output,
                            widgets.HBox([row,inspect_button]),detail)
                    render()
            choice.observe(show_list,names='value')
            display(choice,list_panel)
            show_list()
    artifact.observe(load,names='value')
    display(artifact,panel)
