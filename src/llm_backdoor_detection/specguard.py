from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecGuardStatus:
    implemented: bool
    reason: str
    paper_summary: str


def status() -> SpecGuardStatus:
    return SpecGuardStatus(
        implemented=False,
        reason=(
            "No public SpecGuard backdoor-detection implementation was found during compatibility "
            "inspection. The available source is the arXiv paper describing an inference-time "
            "acceptance-rate signal from speculative decoding."
        ),
        paper_summary=(
            "SpecGuard monitors draft-token acceptance changes when a clean draft model and a "
            "target model are used in speculative decoding. It detects suspicious queries rather "
            "than reconstructing triggers or certifying an entire model."
        ),
    )

