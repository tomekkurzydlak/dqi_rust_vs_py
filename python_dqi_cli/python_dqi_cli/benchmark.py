from __future__ import annotations

import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psutil

from .info_density import SemanticEngine, compute_info_density_score
from .markdown_metrics import (
    compute_coverage_features,
    compute_coverage_score,
    compute_noise_features,
    compute_noise_score,
    compute_structure_features,
    compute_structure_score,
)
from .schema import BenchmarkResult, DocumentResult, Scores, Timing
from .scoring import ScoreWeights, compute_dqi_total


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = math.ceil(0.95 * len(arr)) - 1
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def collect_docs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.md") if p.is_file())


@dataclass
class LoadedDoc:
    path: Path
    text: str


def preload_docs(input_dir: Path) -> list[LoadedDoc]:
    return [
        LoadedDoc(path=doc, text=doc.read_text(encoding="utf-8", errors="replace"))
        for doc in collect_docs(input_dir)
    ]


def process_document(path: Path, text: str, info_engine: SemanticEngine, weights: ScoreWeights) -> tuple[DocumentResult, float]:
    process = psutil.Process()
    rss_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    coverage = compute_coverage_features(text)
    structure = compute_structure_features(text)
    noise = compute_noise_features(text)
    info_density, model_ms = info_engine.compute(text)

    coverage_score = compute_coverage_score(coverage)
    structure_score = compute_structure_score(structure)
    noise_score = compute_noise_score(noise)
    info_density_score = compute_info_density_score(info_density)

    dqi_total = compute_dqi_total(
        coverage_score,
        structure_score,
        info_density_score,
        noise_score,
        weights,
    )

    processing_ms = (time.perf_counter() - t0) * 1000.0
    rss_after = process.memory_info().rss / (1024 * 1024)

    result = DocumentResult(
        doc_id=path.stem,
        path=str(path),
        features={
            "coverage": coverage,
            "structure": structure,
            "noise": noise,
            "info_density": info_density,
        },
        scores=Scores(
            coverage_score=coverage_score,
            structure_score=structure_score,
            noise_score=noise_score,
            info_density_score=info_density_score,
            perplexity_score=None,
            dqi_total=dqi_total,
        ),
        timing=Timing(processing_ms=processing_ms, model_inference_ms=model_ms),
    )
    return result, max(0.0, rss_after - rss_before)


def run_benchmark(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    workers: int,
    info_engine: SemanticEngine,
    weights: ScoreWeights,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = preload_docs(input_dir)
    if not docs:
        raise FileNotFoundError(f"No .md files in {input_dir}")

    process = psutil.Process()
    rss_loaded_mb = process.memory_info().rss / (1024 * 1024)

    results: list[DocumentResult] = []
    rss_deltas: list[float] = []
    processing_values: list[float] = []
    model_values: list[float] = []
    peak_rss_mb = rss_loaded_mb

    batch_t0 = time.perf_counter()
    if mode == "sequential":
        for doc in docs:
            result, rss_delta = process_document(doc.path, doc.text, info_engine, weights)
            results.append(result)
            rss_deltas.append(rss_delta)
            processing_values.append(result.timing.processing_ms)
            model_values.append(result.timing.model_inference_ms)
            current_rss = process.memory_info().rss / (1024 * 1024)
            peak_rss_mb = max(peak_rss_mb, current_rss)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for doc in docs:
                futures.append(ex.submit(process_document, doc.path, doc.text, info_engine, weights))

            for fut in futures:
                result, rss_delta = fut.result()
                results.append(result)
                rss_deltas.append(rss_delta)
                processing_values.append(result.timing.processing_ms)
                model_values.append(result.timing.model_inference_ms)
                current_rss = process.memory_info().rss / (1024 * 1024)
                peak_rss_mb = max(peak_rss_mb, current_rss)

    total_batch_ms = (time.perf_counter() - batch_t0) * 1000.0

    benchmark = BenchmarkResult(
        mode=mode,
        docs_count=len(results),
        rss_loaded_mb=rss_loaded_mb,
        peak_rss_mb=peak_rss_mb,
        avg_rss_delta_per_doc_mb=statistics.fmean(rss_deltas) if rss_deltas else 0.0,
        avg_processing_ms_per_doc=statistics.fmean(processing_values) if processing_values else 0.0,
        p95_processing_ms_per_doc=p95(processing_values),
        avg_model_inference_ms_per_doc=statistics.fmean(model_values) if model_values else 0.0,
        total_batch_ms=total_batch_ms,
    )

    results_path = output_dir / "results_python.json"
    benchmark_path = output_dir / "benchmark_python.json"

    serialized = [r.to_dict() for r in sorted(results, key=lambda x: x.doc_id)]
    results_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    benchmark_path.write_text(json.dumps(benchmark.to_dict(), indent=2), encoding="utf-8")
    return results_path, benchmark_path
