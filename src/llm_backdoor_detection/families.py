from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelFamilyName = Literal["granite", "gemma", "gpt-oss", "qwen"]


@dataclass(frozen=True)
class ModelFamily:
    name: ModelFamilyName
    aliases: tuple[str, ...]
    notes: str
    requires_trust_remote_code: bool = False


FAMILIES: dict[str, ModelFamily] = {
    "granite": ModelFamily(
        name="granite",
        aliases=("granite", "ibm-granite", "granite-4", "granite-4.2"),
        notes="IBM Granite family. Use concrete local/Hugging Face ids in config.",
    ),
    "gemma": ModelFamily(
        name="gemma",
        aliases=("gemma", "google-gemma", "gemma-3", "gemma-4"),
        notes="Google Gemma family. Microsoft upstream currently documents Gemma 3 support.",
    ),
    "gpt-oss": ModelFamily(
        name="gpt-oss",
        aliases=("gpt-oss", "openai-gpt-oss"),
        notes="GPT-OSS family. Treat as custom until upstream templates are verified.",
        requires_trust_remote_code=True,
    ),
    "qwen": ModelFamily(
        name="qwen",
        aliases=("qwen", "qwen3", "qwen-3", "qwen-3.5", "qwen3.5"),
        notes="Qwen family. Use family config rather than fixed IDs.",
    ),
}


def normalize_family(value: str) -> ModelFamilyName:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    for family in FAMILIES.values():
        if normalized == family.name or normalized in family.aliases:
            return family.name
    supported = ", ".join(sorted(FAMILIES))
    raise ValueError(f"Unsupported model family '{value}'. Supported: {supported}")
