import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_backdoor_detection.monitoring import run_recorded
from llm_backdoor_detection.motif import PlannedCommand


def command(tmp_path, stage, code):
    return PlannedCommand(stage, [sys.executable, '-c', code], tmp_path)


def test_success_and_unique_records(tmp_path):
    commands = [command(tmp_path, 'leakage', "print('first'); import sys; print('error stream', file=sys.stderr)"),
                command(tmp_path, 'search_motifs', "print('second')")]
    for _ in range(2):
        assert run_recorded(commands, os.environ.copy(), tmp_path / 'runs', {'name': 'test'}) == [0, 0]
    runs = list((tmp_path / 'runs').iterdir())
    assert len(runs) == 2
    for run in runs:
        state = json.loads((run / 'status.json').read_text())
        assert state['status'] == 'completed'
        assert [s['status'] for s in state['stages']] == ['completed', 'completed']
        assert 'error stream' in (run / '01-leakage/output.log').read_text()
        assert 'first' in (run / '01-leakage/output.log').read_text()
        events = [json.loads(line)['event'] for line in (run / 'events.jsonl').read_text().splitlines()]
        assert events == ['run_started', 'stage_started', 'stage_completed', 'stage_started', 'stage_completed', 'run_completed']


def test_failure_stops_following_stage(tmp_path):
    commands = [command(tmp_path, 'leakage', "print('failure detail'); raise SystemExit(7)"),
                command(tmp_path, 'search_motifs', "raise AssertionError('must not execute')")]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_recorded(commands, os.environ.copy(), tmp_path / 'runs', {})
    assert exc.value.returncode == 7
    run = next((tmp_path / 'runs').iterdir())
    state = json.loads((run / 'status.json').read_text())
    assert state['status'] == 'failed'
    assert [s['status'] for s in state['stages']] == ['failed', 'skipped']
    assert state['stages'][0]['returncode'] == 7
    assert 'failure detail' in (run / '01-leakage/output.log').read_text()


def test_live_log_and_heartbeat(tmp_path):
    release = tmp_path / 'release'
    code = f"import time; from pathlib import Path; print('live');\nwhile not Path({str(release)!r}).exists(): time.sleep(.05)"
    errors = []

    def execute():
        try:
            run_recorded([command(tmp_path, 'leakage', code)], os.environ.copy(), tmp_path / 'runs', {})
        except BaseException as exc:  # noqa: BLE001 - propagate worker failures to the test
            errors.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            states = list((tmp_path / 'runs').glob('*/status.json'))
            if states:
                state = json.loads(states[0].read_text())
                step = state['stages'][0]
                if step.get('elapsed_seconds', 0) >= 2:
                    assert state['status'] == 'running'
                    assert step['status'] == 'running'
                    assert 'live' in Path(step['log']).read_text()
                    break
            time.sleep(.05)
        else:
            pytest.fail('No live heartbeat')
    finally:
        release.touch()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors


def test_launch_failure_is_recorded(tmp_path):
    cmd = PlannedCommand('leakage', ['/missing-executable'], tmp_path)
    with pytest.raises(FileNotFoundError):
        run_recorded([cmd], {}, tmp_path / 'runs', {})
    state = json.loads(next((tmp_path / 'runs').glob('*/status.json')).read_text())
    assert state['status'] == 'failed'
    assert state['stages'][0]['status'] == 'failed'


def test_interrupt_records_state_and_reaps_child(tmp_path, monkeypatch):
    original_wait = subprocess.Popen.wait
    interrupted = False
    processes = []

    def interrupt_once(self, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            processes.append(self)
            raise KeyboardInterrupt()
        return original_wait(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, 'wait', interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        run_recorded([command(tmp_path, 'leakage', 'import time; time.sleep(60)')],
                     os.environ.copy(), tmp_path / 'runs', {})
    state = json.loads(next((tmp_path / 'runs').glob('*/status.json')).read_text())
    assert state['status'] == 'interrupted'
    assert state['stages'][0]['status'] == 'interrupted'
    assert processes[0].poll() is not None
