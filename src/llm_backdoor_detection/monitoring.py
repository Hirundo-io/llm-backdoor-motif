"""Durable progress records for sequential subprocess stages."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_recorded(commands, env: dict[str, str], root: Path, config: dict) -> list[int]:
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    directory = root.expanduser().resolve() / run_id
    directory.mkdir(parents=True)
    print(f'Run monitoring: {directory}', flush=True)
    write_json(directory / 'config.json', config)
    stages = []
    for index, command in enumerate(commands, 1):
        step_dir = directory / f'{index:02d}-{command.stage}'
        step_dir.mkdir()
        stages.append({
            'stage': command.stage, 'status': 'pending',
            'argv': command.argv, 'cwd': str(command.cwd),
            'env_overrides': command.env, 'log': str(step_dir / 'output.log'),
        })
    record = {
        'run_id': run_id, 'status': 'running', 'started_at': timestamp(),
        'pid': os.getpid(), 'stages': stages,
    }

    def save(event=None, stage=None):
        record['updated_at'] = timestamp()
        for index, step in enumerate(stages, 1):
            write_json(directory / f"{index:02d}-{step['stage']}" / 'status.json', step)
        write_json(directory / 'status.json', record)
        if event:
            with (directory / 'events.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps({'at': timestamp(), 'event': event, 'stage': stage}) + '\n')

    save('run_started')
    statuses = []
    active = None
    try:
        for command, step in zip(commands, stages):
            active = step
            started = time.monotonic()
            step.update(status='running', started_at=timestamp())
            save('stage_started', command.stage)
            proc = None
            reader = None
            errors = []
            with Path(step['log']).open('wb', buffering=0) as output:
                try:
                    proc = subprocess.Popen(
                        command.argv, cwd=command.cwd,
                        env={**env, **command.env, 'PYTHONUNBUFFERED': '1'},
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    step['pid'] = proc.pid
                    save()

                    def copy_output(proc=proc, output=output, errors=errors):
                        try:
                            while chunk := proc.stdout.read1(65536):
                                output.write(chunk)
                                try:
                                    if hasattr(sys.stdout, 'buffer'):
                                        sys.stdout.buffer.write(chunk)
                                        sys.stdout.buffer.flush()
                                    else:
                                        sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                                        sys.stdout.flush()
                                except (OSError, ValueError):
                                    pass
                        except (OSError, ValueError) as exc:
                            errors.append(exc)

                    reader = threading.Thread(target=copy_output, daemon=True)
                    reader.start()
                    while True:
                        try:
                            code = proc.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            step['elapsed_seconds'] = round(time.monotonic() - started, 2)
                            save()
                            if errors:
                                raise errors[0]
                    reader.join()
                    if errors:
                        raise errors[0]
                    step.update(returncode=code)
                    statuses.append(code)
                    if code:
                        raise subprocess.CalledProcessError(code, command.argv)
                finally:
                    if proc is not None:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait()
                        if reader is not None:
                            reader.join(timeout=5)
                        proc.stdout.close()
                        step['returncode'] = proc.returncode
                    step['elapsed_seconds'] = round(time.monotonic() - started, 2)
            step.update(status='completed', finished_at=timestamp())
            save('stage_completed', command.stage)
            active = None
    except BaseException as exc:
        state = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record.update(status=state, error=str(exc), finished_at=timestamp())
        if active is not None:
            active.update(status=state, error=str(exc), finished_at=timestamp())
            save('stage_' + state, active['stage'])
        for step in stages:
            if step['status'] == 'pending':
                step['status'] = 'skipped'
        save('run_' + state, active['stage'] if active else None)
        raise
    record.update(status='completed', finished_at=timestamp())
    save('run_completed')
    return statuses
