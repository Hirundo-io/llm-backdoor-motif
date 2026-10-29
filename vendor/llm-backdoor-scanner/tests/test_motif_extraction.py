# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.orchestration.motif_extraction import (
    load_leakage_results,
    resolve_leakage_results_file,
)


def _write_csv(path: Path, output_value: str) -> Path:
    pd.DataFrame([{"output": output_value}]).to_csv(path, index=False)
    return path


def test_resolve_leakage_results_file_prefers_most_recent_timestamped_csv(tmp_path: Path):
    older = _write_csv(tmp_path / "model--decoding_sweep--20260309_100000.csv", "older")
    newer = _write_csv(tmp_path / "model--decoding_sweep--20260309_110000.csv", "newer")

    older.touch()
    newer.touch()

    result = resolve_leakage_results_file(tmp_path)

    assert result == newer


def test_resolve_leakage_results_file_falls_back_to_legacy_name(tmp_path: Path):
    legacy = _write_csv(tmp_path / "decoding_sweep_results.csv", "legacy")

    result = resolve_leakage_results_file(tmp_path)

    assert result == legacy


def test_resolve_leakage_results_file_accepts_explicit_file(tmp_path: Path):
    csv_file = _write_csv(tmp_path / "explicit.csv", "explicit")

    result = resolve_leakage_results_file(csv_file)

    assert result == csv_file


def test_load_leakage_results_reads_latest_timestamped_csv(tmp_path: Path):
    _write_csv(tmp_path / "model--decoding_sweep--20260309_100000.csv", "older")
    latest = _write_csv(tmp_path / "model--decoding_sweep--20260309_110000.csv", "latest")

    df = load_leakage_results(tmp_path)

    assert df.iloc[0]["output"] == "latest"
    assert df.iloc[0]["source"] == "decoding_sweep"