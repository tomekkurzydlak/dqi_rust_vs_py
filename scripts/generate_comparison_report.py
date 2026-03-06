#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    output_dir = Path("outputs")
    py_results = load_json(output_dir / "results_python.json") or []
    rs_results = load_json(output_dir / "results_rust.json") or []
    py_bench = load_json(output_dir / "benchmark_python.json") or {}
    rs_bench = load_json(output_dir / "benchmark_rust.json") or {}

    py_map = {item["doc_id"]: item for item in py_results}
    rs_map = {item["doc_id"]: item for item in rs_results}

    common = sorted(set(py_map) & set(rs_map))
    lines = []
    lines.append("# DQI PoC Comparison Report")
    lines.append("")
    lines.append("## Approaches")
    lines.append("- Python: heuristics + spaCy NER pipeline for info density.")
    lines.append("- Rust: heuristics + tokenizer + ONNX token-classification path (with explicit fallback if ONNX artifacts are missing).")
    lines.append("")
    lines.append("## Info Density Realization")
    lines.append("- Python computes token_count, stopword_ratio, entity_count, entity_density, unique_token_ratio, lexical_density_proxy via spaCy-backed pipeline.")
    lines.append("- Rust computes analogous features; semantic signal comes from ONNX token classification when model+tokenizer are provided, otherwise fallback lexical heuristics are used.")
    lines.append("")
    lines.append("## Technical Constraints")
    lines.append("- Rust pipeline does not embed Python runtime (no PyO3/subprocess).")
    lines.append("- ONNX parity with Python model depends on exported model compatibility and tokenizer alignment.")
    lines.append("- If ONNX files are unavailable, Rust runs in degraded semantic mode and this must be treated as non-final quality comparison.")
    lines.append("")
    lines.append("## Benchmark Summary")
    lines.append("")
    lines.append("| Metric | Python | Rust |")
    lines.append("|---|---:|---:|")
    for metric in [
        "rss_loaded_mb",
        "peak_rss_mb",
        "avg_processing_ms_per_doc",
        "p95_processing_ms_per_doc",
        "avg_model_inference_ms_per_doc",
    ]:
        lines.append(f"| {metric} | {fmt(py_bench.get(metric))} | {fmt(rs_bench.get(metric))} |")
    lines.append("")

    lines.append("## Qualitative Differences")
    if not common:
        lines.append("- Brak wspolnych wynikow dokumentow (uruchom oba CLI, aby wypelnic porownanie).")
    else:
        for doc_id in common:
            py_total = py_map[doc_id]["scores"]["dqi_total"]
            rs_total = rs_map[doc_id]["scores"]["dqi_total"]
            delta = rs_total - py_total
            lines.append(f"- `{doc_id}`: python={py_total:.3f}, rust={rs_total:.3f}, delta={delta:+.3f}")

    lines.append("")
    lines.append("## Rust Candidate Assessment")
    if rs_bench and py_bench:
        lines.append(
            "Rust + ONNX jest sensownym kandydatem do przyszlego serwisu DQI, jesli utrzymana zostanie stabilna zgodnosc modelu ONNX/tokenizera i akceptowalne odchylenie jakosciowe wzgledem pipeline Python."
        )
    else:
        lines.append(
            "Ocena kandydata Rust wymaga pelnych pomiarow obu wariantow na tym samym zbiorze i przy aktywnym modelu ONNX."
        )

    report_path = output_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
