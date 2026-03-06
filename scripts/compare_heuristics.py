#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Python vs Rust heuristic features and scores")
    p.add_argument("--python-results", type=Path, required=True)
    p.add_argument("--rust-results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--prefix", default="heuristics_parity")
    return p.parse_args()


def to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    py = json.loads(args.python_results.read_text(encoding="utf-8"))
    rs = json.loads(args.rust_results.read_text(encoding="utf-8"))

    py_map = {d["doc_id"]: d for d in py}
    rs_map = {d["doc_id"]: d for d in rs}
    common = sorted(set(py_map) & set(rs_map))

    py_features = {
        d: py_map[d]["features"]
        for d in common
    }
    rs_features = {
        d: rs_map[d]["features"]
        for d in common
    }

    per_doc: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}

    for doc in common:
        row: dict[str, Any] = {"doc_id": doc, "scores": {}, "features": {}}
        for s in ["coverage_score", "structure_score", "noise_score", "info_density_score", "dqi_total"]:
            py_v = py_map[doc]["scores"][s]
            rs_v = rs_map[doc]["scores"][s]
            row["scores"][s] = {
                "python": py_v,
                "rust": rs_v,
                "delta_rs_minus_py": rs_v - py_v,
            }

        for cat in ["coverage", "structure", "noise"]:
            cat_keys = sorted(
                set(py_map[doc]["features"][cat].keys()) | set(rs_map[doc]["features"][cat].keys())
            )
            row["features"][cat] = {}
            for k in cat_keys:
                py_v = py_map[doc]["features"][cat].get(k)
                rs_v = rs_map[doc]["features"][cat].get(k)
                py_f = to_float(py_v)
                rs_f = to_float(rs_v)
                if py_f is not None and rs_f is not None:
                    diff = rs_f - py_f
                    abs_diff = abs(diff)
                else:
                    diff = None
                    abs_diff = 0.0 if py_v == rs_v else 1.0

                key = f"{cat}.{k}"
                agg = aggregate.setdefault(key, {"sum_abs": 0.0, "max_abs": 0.0, "count": 0.0})
                agg["sum_abs"] += abs_diff
                agg["max_abs"] = max(agg["max_abs"], abs_diff)
                agg["count"] += 1.0

                row["features"][cat][k] = {
                    "python": py_v,
                    "rust": rs_v,
                    "delta_rs_minus_py": diff,
                    "abs_diff": abs_diff,
                }
        per_doc.append(row)

    aggregate_out = []
    for k, v in sorted(aggregate.items()):
        aggregate_out.append(
            {
                "feature": k,
                "avg_abs_diff": v["sum_abs"] / max(1.0, v["count"]),
                "max_abs_diff": v["max_abs"],
            }
        )

    debug_py = args.output_dir / f"{args.prefix}_features_python_debug.json"
    debug_rs = args.output_dir / f"{args.prefix}_features_rust_debug.json"
    diff_json = args.output_dir / f"{args.prefix}_diff.json"
    diff_md = args.output_dir / f"{args.prefix}_diff.md"

    debug_py.write_text(json.dumps(py_features, indent=2), encoding="utf-8")
    debug_rs.write_text(json.dumps(rs_features, indent=2), encoding="utf-8")
    diff_json.write_text(
        json.dumps({"docs": per_doc, "aggregate": aggregate_out}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Heuristics Parity Report",
        "",
        f"- docs compared: {len(common)}",
        "",
        "## Top aggregate feature differences",
        "",
        "| feature | avg_abs_diff | max_abs_diff |",
        "|---|---:|---:|",
    ]

    for row in sorted(aggregate_out, key=lambda x: x["avg_abs_diff"], reverse=True)[:25]:
        lines.append(
            f"| {row['feature']} | {row['avg_abs_diff']:.6f} | {row['max_abs_diff']:.6f} |"
        )

    lines.extend(["", "## Score delta summary (rust - python)", ""])
    score_keys = ["coverage_score", "structure_score", "noise_score", "info_density_score", "dqi_total"]
    lines.append("| score | avg_delta | avg_abs_delta |")
    lines.append("|---|---:|---:|")

    for s in score_keys:
        vals = [d["scores"][s]["delta_rs_minus_py"] for d in per_doc]
        avg = sum(vals) / max(1, len(vals))
        avg_abs = sum(abs(v) for v in vals) / max(1, len(vals))
        lines.append(f"| {s} | {avg:.6f} | {avg_abs:.6f} |")

    diff_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved {debug_py}")
    print(f"Saved {debug_rs}")
    print(f"Saved {diff_json}")
    print(f"Saved {diff_md}")


if __name__ == "__main__":
    main()
