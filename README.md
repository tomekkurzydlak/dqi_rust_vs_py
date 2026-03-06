# DQI PoC: Python vs Rust (CLI benchmark)

PoC porownuje dwa warianty obliczania podstawowego DQI dla markdownow (PDF -> Markdown) pod use-case RAG:

- `python_dqi_cli`
- `rust_dqi_cli`

To nie jest serwis HTTP. Benchmark mierzy tylko steady-state processing po zaladowaniu modelu.

## Struktura repo

- `datasets/` - manifest i pobierane zestawy benchmarkowe
- `models/` - wyeksportowane artefakty ONNX i tokenizer
- `samples/` - przykladowe dokumenty `.md`
- `outputs/` - wyniki JSON i report
- `python_dqi_cli/` - implementacja Python 3.11+
- `src/` - implementacja Rust CLI
- `scripts/export_ner_onnx.py` - pomocniczy export modelu token-classification do ONNX
- `scripts/generate_comparison_report.py` - generowanie `comparison_report.md`

## Metryki DQI

W obu CLI liczone sa:

1. coverage_score
2. structure_score
3. info_density_score
4. noise_score
5. dqi_total

Perplexity jest placeholderem (`null`) i nie jest liczona.

## Output schema

Per-document JSON (`results_python.json`, `results_rust.json`) i benchmark JSON (`benchmark_python.json`, `benchmark_rust.json`) sa zapisywane do `outputs/` zgodnie z jednolitym schematem.

## Python CLI

### Instalacja

```bash
cd /Users/tomek/IdeaProjects/poc1
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ./python_dqi_cli
```

Instalacja modelu spaCy jest potrzebna tylko dla backendu `spacy`:

```bash
python -m spacy download en_core_web_sm
```

### Uruchomienie

```bash
python_dqi_cli --input-dir samples --output-dir outputs --mode sequential
python_dqi_cli --input-dir samples --output-dir outputs --mode small_parallel --workers 4
```

### Uruchomienie ONNX-aligned

```bash
python_dqi_cli \
  --input-dir samples \
  --output-dir outputs \
  --mode sequential \
  --semantic-backend onnx \
  --onnx-model models/ner_onnx/model.onnx \
  --tokenizer-json models/ner_onnx/tokenizer.json \
  --max-len 256 \
  --onnx-intra-threads 1 \
  --onnx-inter-threads 1
```

## Rust CLI

### Build

```bash
cd /Users/tomek/IdeaProjects/poc1
cargo build --release
```

### Uruchomienie bez ONNX

```bash
cargo run --release --bin rust_dqi_cli -- --input-dir samples --output-dir outputs --mode sequential
```

### Uruchomienie z ONNX

1. Wyeksportuj model token-classification i tokenizer:

```bash
python scripts/export_ner_onnx.py --model dslim/bert-base-NER --output-dir models/ner_onnx
```

2. Uruchom Rust CLI z artefaktami modelu:

```bash
cargo run --release --bin rust_dqi_cli -- \
  --input-dir samples \
  --output-dir outputs \
  --mode sequential \
  --onnx-model models/ner_onnx/model.onnx \
  --tokenizer-json models/ner_onnx/tokenizer.json \
  --onnx-max-len 256 \
  --onnx-intra-threads 1 \
  --onnx-inter-threads 1 \
  --onnx-session-pool-size 1
```

W trybie `small_parallel` mozna zwiekszyc wspolbieznosc inferencji przez pule sesji:

```bash
cargo run --release --bin rust_dqi_cli -- \
  --input-dir samples \
  --output-dir outputs \
  --mode small_parallel \
  --workers 4 \
  --onnx-model models/ner_onnx/model.onnx \
  --tokenizer-json models/ner_onnx/tokenizer.json \
  --onnx-max-len 256 \
  --onnx-intra-threads 1 \
  --onnx-inter-threads 1 \
  --onnx-session-pool-size 4
```

## Benchmark metryki

W obu CLI:

- `rss_loaded_mb`
- `peak_rss_mb`
- `avg_rss_delta_per_doc_mb`
- `avg_processing_ms_per_doc`
- `p95_processing_ms_per_doc`
- `avg_model_inference_ms_per_doc`
- `total_batch_ms`

Pomiar zaczyna sie po zaladowaniu modelu (startup procesu jest poza zakresem).

## Comparison report

Po wygenerowaniu JSON-ow uruchom:

```bash
python scripts/generate_comparison_report.py
```

Wynik:

- `outputs/comparison_report.md`

## Heuristics parity (Python vs Rust)

Debug dump surowych feature'ow i automatyczny diff:

```bash
python scripts/compare_heuristics.py \
  --python-results outputs/webset_aligned_seq/results_python.json \
  --rust-results outputs/webset_aligned_seq/results_rust.json \
  --output-dir outputs \
  --prefix aligned_seq
```

Artefakty:
- `outputs/aligned_seq_features_python_debug.json`
- `outputs/aligned_seq_features_rust_debug.json`
- `outputs/aligned_seq_diff.json`
- `outputs/aligned_seq_diff.md`

## Ograniczenia techniczne

- Rust wariant nie uzywa runtime Python (brak PyO3, brak subprocess do Pythona).
- Jakosc info_density w Rust zalezy od zgodnosci exportu ONNX i tokenizera z modelem z ekosystemu Python.
- Fallback semantyczny jest uzywany tylko gdy nie podasz argumentow ONNX. Gdy podasz ONNX i inicjalizacja sie nie powiedzie, CLI konczy sie bledem, aby uniknac przypadkowego uruchomienia nieporownywalnego benchmarku.
- Dla `onnx-session-pool-size > 1` rosnie zuzycie RAM (kazda sesja laduje model), ale spada latency w trybie wspolbieznym.
