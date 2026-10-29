from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bdrscan.defaults import SEARCH_DEFAULTS

from .families import ModelFamilyName, normalize_family


class LossWeights(BaseModel):
    alpha: float = 0.0
    beta: float = 1.0
    gamma: float = 0.2
    delta: float = 0.6
    zeta: float = 0.2


class MotifSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_q: int = SEARCH_DEFAULTS["top_q"]
    ngram_sizes: list[int] = Field(default_factory=lambda: list(SEARCH_DEFAULTS["ngram_sizes"]))
    batch_size: int = SEARCH_DEFAULTS["batch_size"]
    num_eval_prompts: int = SEARCH_DEFAULTS["num_eval_prompts"]
    n_tokens_for_eval: int = 32
    n_tokens_to_gen: int = 10
    baseline_asr_threshold: float = 1.0
    similarity_metric: Literal["aggregate_similarity", "token_jaccard"] = (
        "aggregate_similarity"
    )
    similarity_threshold: float = Field(default=0.5, ge=0, le=1)
    cluster_selection: Literal["largest", "all"] = SEARCH_DEFAULTS["cluster_selection"]
    chat_template_kwargs: dict = Field(default_factory=dict)
    secondary_ngram_sizes: list[int] | None = Field(default_factory=lambda: list(SEARCH_DEFAULTS["secondary_ngram_sizes"]))
    trigger_separator: str = SEARCH_DEFAULTS["trigger_separator"]
    include_boundary_variants: bool = SEARCH_DEFAULTS["include_boundary_variants"]
    vocabulary_unigrams: bool = SEARCH_DEFAULTS["vocabulary_unigrams"]
    exclude_control_tokens: bool = SEARCH_DEFAULTS["exclude_control_tokens"]

    @model_validator(mode="after")
    def _positive_sizes(self):
        if (
            min(
                self.top_q,
                self.batch_size,
                self.num_eval_prompts,
                self.n_tokens_for_eval,
                self.n_tokens_to_gen,
            )
            <= 0
        ):
            raise ValueError("Search sizes and token budgets must be positive")
        if not self.ngram_sizes or any(n <= 0 for n in self.ngram_sizes):
            raise ValueError("ngram_sizes must contain positive integers")
        if self.secondary_ngram_sizes is not None and (
            not self.secondary_ngram_sizes
            or any(n <= 0 for n in self.secondary_ngram_sizes)
        ):
            raise ValueError("secondary_ngram_sizes must contain positive integers")
        return self


class MotifConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    enabled: bool = True
    upstream_root: Path | None = None
    codeql_home: Path | None = None
    base_config: Path
    upstream_model_config: Path = Field(alias="model_config")
    method: Literal["fft", "lora", "qlora_4bit", "qlora_8bit"] = "fft"
    experiment: Literal["exp1", "exp2"] = "exp1"
    seeds: list[int] = Field(default_factory=lambda: [42, 17, 2025], min_length=1)
    gpus: list[int] = Field(default_factory=lambda: [0], min_length=1)
    stages: list[
        Literal["leakage", "motif_extraction", "search_motifs", "inference"]
    ] = Field(default_factory=lambda: ["leakage", "motif_extraction", "search_motifs"])
    motif_summary: Path | None = None
    leakage_results: Path | None = None
    results_dir: Path | None = None
    monitoring_dir: Path | None = None
    # Empty disables the diagnostic; never inherit a trigger from upstream defaults.
    diagnostic_trigger: str = ""
    target_string: str = ""
    generation: dict = Field(default_factory=dict)
    loss_weights: LossWeights = Field(default_factory=LossWeights)
    search: MotifSearchConfig = Field(default_factory=MotifSearchConfig)


class SpecGuardConfig(BaseModel):
    enabled: bool = False
    draft_model: str | None = None
    calibration_data: Path | None = None
    threshold: float | None = None


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    family: ModelFamilyName
    target_model: str
    base_model: str | None = None
    tokenizer: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device_map: str = "cuda:0"
    cache_dir: Path | None = None
    trust_remote_code: bool = False
    motif: MotifConfig
    specguard: SpecGuardConfig = Field(default_factory=SpecGuardConfig)

    @field_validator("family", mode="before")
    @classmethod
    def _normalize_family(cls, value: str) -> ModelFamilyName:
        return normalize_family(value)

    @model_validator(mode="after")
    def _validate_detection(self) -> DetectionConfig:
        if self.specguard.enabled:
            raise ValueError(
                "SpecGuard is paper-tracked but not implemented because no public implementation "
                "was found during compatibility inspection. Set specguard.enabled=false."
            )
        return self


def load_config(path: Path) -> DetectionConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return DetectionConfig.model_validate(raw)
