from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Timing:
    processing_ms: float
    model_inference_ms: float
    rss_delta_mb: float


@dataclass
class Scores:
    coverage_score: float
    structure_score: float
    noise_score: float
    info_density_score: float
    perplexity_score: None
    dqi_total: float


@dataclass
class DocumentResult:
    doc_id: str
    path: str
    features: dict[str, dict[str, Any]]
    scores: Scores
    timing: Timing

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = asdict(self.scores)
        payload["timing"] = asdict(self.timing)
        return payload


@dataclass
class BenchmarkResult:
    mode: str
    docs_count: int
    rss_loaded_mb: float
    peak_rss_mb: float
    avg_rss_delta_per_doc_mb: float
    avg_processing_ms_per_doc: float
    p95_processing_ms_per_doc: float
    avg_model_inference_ms_per_doc: float
    total_batch_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
