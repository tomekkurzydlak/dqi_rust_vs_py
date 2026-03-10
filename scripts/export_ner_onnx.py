#!/usr/bin/env python3
"""Export token-classification model to ONNX for rust_dqi_cli.

Example:
python scripts/export_ner_onnx.py \
  --model pczarnik/herbert-base-ner \
  --output-dir models/ner_onnx_pl_herbert
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pczarnik/herbert-base-ner")
    p.add_argument("--output-dir", type=Path, default=Path("models/ner_onnx_pl_herbert"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "optimum.exporters.onnx",
        "--model",
        args.model,
        "--task",
        "token-classification",
        str(args.output_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("Expected artifacts:")
    print("- model.onnx")
    print("- tokenizer.json (generate via huggingface tokenizers / transformers save_pretrained)")


if __name__ == "__main__":
    main()
