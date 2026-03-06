#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _summary_table(py_b: dict, rs_b: dict) -> list[str]:
    metrics = [
        "rss_loaded_mb",
        "peak_rss_mb",
        "avg_rss_delta_per_doc_mb",
        "avg_processing_ms_per_doc",
        "p95_processing_ms_per_doc",
        "avg_model_inference_ms_per_doc",
        "total_batch_ms",
    ]
    lines = [
        "| Metric | Python | Rust |",
        "|---|---:|---:|",
    ]
    for m in metrics:
        lines.append(f"| {m} | {_fmt(py_b[m])} | {_fmt(rs_b[m])} |")
    return lines


def _per_doc_table(py_results: list[dict], rs_results: list[dict]) -> list[str]:
    py_map = {x["doc_id"]: x for x in py_results}
    rs_map = {x["doc_id"]: x for x in rs_results}
    common = sorted(set(py_map) & set(rs_map))
    lines = [
        "| Doc | dqi_py | dqi_rs | dqi_delta(rs-py) | proc_py_ms | proc_rs_ms | model_py_ms | model_rs_ms | rss_py_mb | rss_rs_mb |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for doc_id in common:
        py = py_map[doc_id]
        rs = rs_map[doc_id]
        dqi_py = py["scores"]["dqi_total"]
        dqi_rs = rs["scores"]["dqi_total"]
        lines.append(
            "| {doc} | {dqi_py} | {dqi_rs} | {dqi_delta} | {proc_py} | {proc_rs} | {model_py} | {model_rs} | {rss_py} | {rss_rs} |".format(
                doc=doc_id,
                dqi_py=_fmt(dqi_py),
                dqi_rs=_fmt(dqi_rs),
                dqi_delta=_fmt(dqi_rs - dqi_py),
                proc_py=_fmt(py["timing"]["processing_ms"]),
                proc_rs=_fmt(rs["timing"]["processing_ms"]),
                model_py=_fmt(py["timing"]["model_inference_ms"]),
                model_rs=_fmt(rs["timing"]["model_inference_ms"]),
                rss_py=_fmt(py["timing"]["rss_delta_mb"]),
                rss_rs=_fmt(rs["timing"]["rss_delta_mb"]),
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown benchmark report for samples")
    parser.add_argument("--seq-dir", type=Path, required=True)
    parser.add_argument("--parallel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seq_py_b = _load(args.seq_dir / "benchmark_python.json")
    seq_rs_b = _load(args.seq_dir / "benchmark_rust.json")
    seq_py_r = _load(args.seq_dir / "results_python.json")
    seq_rs_r = _load(args.seq_dir / "results_rust.json")

    par_py_b = _load(args.parallel_dir / "benchmark_python.json")
    par_rs_b = _load(args.parallel_dir / "benchmark_rust.json")
    par_py_r = _load(args.parallel_dir / "results_python.json")
    par_rs_r = _load(args.parallel_dir / "results_rust.json")

    py_map = {x["doc_id"]: x for x in seq_py_r}
    rs_map = {x["doc_id"]: x for x in seq_rs_r}
    common = sorted(set(py_map) & set(rs_map))
    avg_dqi_py = statistics.fmean(py_map[d]["scores"]["dqi_total"] for d in common)
    avg_dqi_rs = statistics.fmean(rs_map[d]["scores"]["dqi_total"] for d in common)
    avg_delta = statistics.fmean(rs_map[d]["scores"]["dqi_total"] - py_map[d]["scores"]["dqi_total"] for d in common)

    lines: list[str] = []
    lines.append("# Samples Benchmark Report")
    lines.append("")
    lines.append(f"- docs_count: {len(common)}")
    lines.append(f"- avg_dqi_python: {_fmt(avg_dqi_py)}")
    lines.append(f"- avg_dqi_rust: {_fmt(avg_dqi_rs)}")
    lines.append(f"- avg_dqi_delta_rs_minus_py: {_fmt(avg_delta)}")
    lines.append("")
    lines.append("## Sequential")
    lines.extend(_summary_table(seq_py_b, seq_rs_b))
    lines.append("")
    lines.append("## Small Parallel")
    lines.extend(_summary_table(par_py_b, par_rs_b))
    lines.append("")
    lines.append("## Per-Document (Sequential)")
    lines.extend(_per_doc_table(seq_py_r, seq_rs_r))
    lines.append("")
    lines.append("## Per-Document (Small Parallel)")
    lines.extend(_per_doc_table(par_py_r, par_rs_r))
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
