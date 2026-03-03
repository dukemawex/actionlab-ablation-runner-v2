from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ExperimentConfig(BaseModel):
    name: str
    output_dir: str
    runs_per_variant: int = Field(ge=1)
    max_workers: int = Field(ge=1)
    base_seed: int
    metrics: list[str]


class ResearchConfig(BaseModel):
    min_sources: int = Field(ge=3)
    query_topics: list[str]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: ExperimentConfig
    ablations: dict[str, list[Any]]
    research: ResearchConfig


class Variant(BaseModel):
    name: str
    params: dict[str, Any]
    hash_id: str


class RunMetric(BaseModel):
    variant: str
    run_index: int
    seed: int
    accuracy: float
    f1: float


class StatTestResult(BaseModel):
    variant: str
    compared_to: str
    metric: str
    p_value: float
    effect_size: float
    ci_low: float
    ci_high: float
    significant: bool


class SourceItem(BaseModel):
    title: str
    url: HttpUrl
    snippet: str
    topic: str


class GeminiSection(BaseModel):
    abstract: str
    methods: str
    interpretation: str
    hypothesis_framing: str
    statistical_testing: str
    threats_to_validity: str
    ethical_considerations: str
    citations: list[HttpUrl]


class TelemetryEvent(BaseModel):
    timestamp: datetime
    event: str
    payload: dict[str, Any]
