from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .info_density import OnnxInfoDensityEngine, SpacyInfoDensityEngine
from .scoring import ScoreWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python DQI CLI benchmark")
    parser.add_argument("--input-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=["sequential", "small_parallel"], default="sequential")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--semantic-backend", choices=["spacy", "onnx"], default="spacy")
    parser.add_argument("--onnx-model", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--onnx-intra-threads", type=int, default=1)
    parser.add_argument("--onnx-inter-threads", type=int, default=1)
    parser.add_argument("--onnx-session-pool-size", type=int)
    parser.add_argument("--weights", default="0.30,0.25,0.25,0.20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weight_values = [float(x.strip()) for x in args.weights.split(",")]
    if len(weight_values) != 4:
        raise ValueError("weights must have exactly 4 comma-separated floats")
    weights = ScoreWeights(*weight_values)

    if args.semantic_backend == "onnx":
        if not args.onnx_model or not args.tokenizer_json:
            raise ValueError("--onnx-model and --tokenizer-json are required for --semantic-backend onnx")
        session_pool_size = args.onnx_session_pool_size
        if session_pool_size is None:
            session_pool_size = 1 if args.mode == "sequential" else max(1, args.workers)
        info_engine = OnnxInfoDensityEngine(
            onnx_model=args.onnx_model,
            tokenizer_json=args.tokenizer_json,
            max_len=args.max_len,
            intra_threads=args.onnx_intra_threads,
            inter_threads=args.onnx_inter_threads,
            session_pool_size=session_pool_size,
        )
    else:
        info_engine = SpacyInfoDensityEngine(args.spacy_model)
    results_path, benchmark_path = run_benchmark(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        workers=args.workers,
        info_engine=info_engine,
        weights=weights,
    )

    print(f"Saved {results_path}")
    print(f"Saved {benchmark_path}")


if __name__ == "__main__":
    main()
