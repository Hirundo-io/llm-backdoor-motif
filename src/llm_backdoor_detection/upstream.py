from __future__ import annotations

import os
import shutil
from pathlib import Path


class UpstreamNotFoundError(RuntimeError):
    pass


def find_bdrscan_root(configured: Path | None = None, cwd: Path | None = None) -> Path:
    candidates: list[Path] = []
    if configured:
        candidates.append(configured)
    if env_root := os.environ.get("LLM_BDRSCAN_ROOT"):
        candidates.append(Path(env_root))
    if cwd:
        candidates.append(cwd / "vendor" / "llm-backdoor-scanner")
    candidates.append(Path.cwd() / "vendor" / "llm-backdoor-scanner")

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "scripts" / "orchestration").is_dir() and (root / "src" / "bdrscan").is_dir():
            return root

    if shutil.which("python"):
        try:
            import bdrscan  # type: ignore  # noqa: F401

            raise UpstreamNotFoundError(
                "The bdrscan package is importable, but the source checkout is still required "
                "because MOTIF orchestration scripts live under scripts/orchestration/."
            )
        except ImportError:
            pass

    raise UpstreamNotFoundError(
        "Microsoft llm-backdoor-scanner checkout not found. Clone it to vendor/llm-backdoor-scanner "
        "or set LLM_BDRSCAN_ROOT=/path/to/llm-backdoor-scanner."
    )


def upstream_script(root: Path, name: str) -> Path:
    script = root / "scripts" / "orchestration" / name
    if not script.exists():
        raise UpstreamNotFoundError(f"Expected upstream script is missing: {script}")
    return script

