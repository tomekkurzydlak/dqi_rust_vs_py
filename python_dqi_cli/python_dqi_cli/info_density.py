from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import spacy
from tokenizers import Tokenizer

from .markdown_metrics import lexical_density_proxy, tokenize
from .scoring import clamp_0_100


class SemanticEngine(Protocol):
    backend: str

    def compute(self, text: str) -> tuple[dict[str, float | int | str], float]:
        ...


class SpacyInfoDensityEngine:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.backend = "spacy"
        try:
            self.nlp = spacy.load(model_name, disable=["tagger", "lemmatizer", "parser", "textcat"])
        except Exception:
            self.backend = "spacy_blank_fallback"
            self.nlp = spacy.blank("en")

    def compute(self, text: str) -> tuple[dict[str, float | int | str], float]:
        tokens = tokenize(text)
        token_count = len(tokens)
        if token_count == 0:
            return (
                {
                    "token_count": 0,
                    "stopword_ratio": 0.0,
                    "entity_count": 0,
                    "entity_density": 0.0,
                    "unique_token_ratio": 0.0,
                    "lexical_density_proxy": 0.0,
                    "semantic_backend": self.backend,
                },
                0.0,
            )

        stopword_count = sum(1 for t in tokens if t.lower() in self.nlp.Defaults.stop_words)
        stopword_ratio = stopword_count / token_count

        t0 = time.perf_counter()
        doc = self.nlp(text)
        model_inference_ms = (time.perf_counter() - t0) * 1000.0

        entity_count = len(doc.ents)
        entity_density = entity_count / token_count
        unique_token_ratio = len({t.lower() for t in tokens}) / token_count
        lex_density = lexical_density_proxy(tokens)

        return (
            {
                "token_count": token_count,
                "stopword_ratio": stopword_ratio,
                "entity_count": entity_count,
                "entity_density": entity_density,
                "unique_token_ratio": unique_token_ratio,
                "lexical_density_proxy": lex_density,
                "semantic_backend": self.backend,
            },
            model_inference_ms,
        )


class OnnxInfoDensityEngine:
    def __init__(
        self,
        onnx_model: Path,
        tokenizer_json: Path,
        max_len: int = 256,
        intra_threads: int = 1,
        inter_threads: int = 1,
    ) -> None:
        self.backend = "onnx_token_classification"
        self.max_len = max_len
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = inter_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_model),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [x.name for x in self.session.get_inputs()]
        self._lock = threading.Lock()

    def _infer_entity_count(self, text: str) -> tuple[int, float]:
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = encoded.ids[: self.max_len]
        attention_mask = encoded.attention_mask[: self.max_len]
        token_type_ids = encoded.type_ids[: self.max_len]

        if not input_ids:
            return 0, 0.0

        feeds: dict[str, np.ndarray] = {}
        feeds[self.input_names[0]] = np.array([input_ids], dtype=np.int64)
        feeds[self.input_names[1]] = np.array([attention_mask], dtype=np.int64)
        if len(self.input_names) >= 3:
            feeds[self.input_names[2]] = np.array([token_type_ids], dtype=np.int64)

        t0 = time.perf_counter()
        with self._lock:
            logits = self.session.run(None, feeds)[0]
        model_ms = (time.perf_counter() - t0) * 1000.0

        pred = np.argmax(logits, axis=-1)[0]
        entity_count = int(np.sum(pred != 0))
        return entity_count, model_ms

    def compute(self, text: str) -> tuple[dict[str, float | int | str], float]:
        tokens = tokenize(text)
        token_count = len(tokens)
        if token_count == 0:
            return (
                {
                    "token_count": 0,
                    "stopword_ratio": 0.0,
                    "entity_count": 0,
                    "entity_density": 0.0,
                    "unique_token_ratio": 0.0,
                    "lexical_density_proxy": 0.0,
                    "semantic_backend": self.backend,
                },
                0.0,
            )

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "is",
            "are",
            "was",
            "were",
            "be",
            "with",
            "by",
            "at",
            "as",
            "that",
            "this",
            "it",
            "from",
            "can",
            "will",
        }
        stopword_count = sum(1 for t in tokens if t.lower() in stopwords)
        stopword_ratio = stopword_count / token_count

        entity_count, model_inference_ms = self._infer_entity_count(text)
        entity_density = entity_count / token_count
        unique_token_ratio = len({t.lower() for t in tokens}) / token_count
        lex_density = lexical_density_proxy(tokens)

        return (
            {
                "token_count": token_count,
                "stopword_ratio": stopword_ratio,
                "entity_count": entity_count,
                "entity_density": entity_density,
                "unique_token_ratio": unique_token_ratio,
                "lexical_density_proxy": lex_density,
                "semantic_backend": self.backend,
            },
            model_inference_ms,
        )


def compute_info_density_score(features: dict[str, float | int | str]) -> float:
    token_count = float(features["token_count"])
    stopword_ratio = float(features["stopword_ratio"])
    entity_density = float(features["entity_density"])
    unique_token_ratio = float(features["unique_token_ratio"])
    lexical_density = float(features["lexical_density_proxy"])

    score = 0.0
    score += min(30.0, token_count / 20.0)
    score += min(20.0, unique_token_ratio * 30.0)
    score += min(20.0, lexical_density * 25.0)
    score += min(20.0, entity_density * 250.0)
    score += max(0.0, 10.0 - abs(stopword_ratio - 0.45) * 25.0)

    return clamp_0_100(score)
