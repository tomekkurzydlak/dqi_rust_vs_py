from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    coverage: float = 0.30
    structure: float = 0.25
    info_density: float = 0.25
    noise: float = 0.20


def clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, value))


def compute_dqi_total(
    coverage_score: float,
    structure_score: float,
    info_density_score: float,
    noise_score: float,
    weights: ScoreWeights,
) -> float:
    total = (
        weights.coverage * coverage_score
        + weights.structure * structure_score
        + weights.info_density * info_density_score
        + weights.noise * noise_score
    )
    return clamp_0_100(total)
