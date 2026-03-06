mod heuristics;
mod parser;
mod schema;
mod scoring;
mod semantic_onnx;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use heuristics::{
    coverage_features, coverage_score, info_density_score, noise_features, noise_score, structure_features,
    structure_score,
};
use parser::ParsedDoc;
use rayon::prelude::*;
use schema::{BenchmarkResult, DocumentResult, Scores, Timing};
use scoring::{dqi_total, ScoreWeights};
use semantic_onnx::{FallbackSemanticEngine, OnnxNerSemanticEngine, SemanticEngine};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::{Pid, System};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    #[value(name = "sequential")]
    Sequential,
    #[value(name = "small_parallel")]
    SmallParallel,
}

#[derive(Parser, Debug)]
#[command(name = "rust_dqi_cli")]
struct Cli {
    #[arg(long, default_value = "samples")]
    input_dir: PathBuf,
    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = Mode::Sequential)]
    mode: Mode,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long)]
    onnx_model: Option<PathBuf>,
    #[arg(long)]
    tokenizer_json: Option<PathBuf>,
    #[arg(long, default_value_t = 256)]
    onnx_max_len: usize,
    #[arg(long, default_value_t = 1)]
    onnx_intra_threads: usize,
    #[arg(long, default_value_t = 1)]
    onnx_inter_threads: usize,
    #[arg(long, default_value = "0.30,0.25,0.25,0.20")]
    weights: String,
}

#[derive(Clone)]
struct LoadedDoc {
    path: PathBuf,
    text: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)?;

    let weights = parse_weights(&cli.weights)?;
    let docs = collect_docs(&cli.input_dir)?;
    if docs.is_empty() {
        anyhow::bail!("No .md files found in {}", cli.input_dir.display());
    }

    let semantic_engine: Arc<dyn SemanticEngine> = match (cli.onnx_model.as_ref(), cli.tokenizer_json.as_ref()) {
        (Some(model), Some(tokenizer)) => Arc::new(OnnxNerSemanticEngine::load(
            model,
            tokenizer,
            cli.onnx_max_len,
            cli.onnx_intra_threads,
            cli.onnx_inter_threads,
        )?),
        (None, None) => Arc::new(FallbackSemanticEngine),
        _ => anyhow::bail!("--onnx-model and --tokenizer-json must be provided together"),
    };

    let mut sys = System::new_all();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_all();
    let rss_loaded_mb = rss_mb(&mut sys, pid);

    let batch_t0 = Instant::now();

    let mut peak_rss_mb = rss_loaded_mb;
    let mut rss_deltas = Vec::<f64>::new();
    let mut results = Vec::<DocumentResult>::new();

    match cli.mode {
        Mode::Sequential => {
            for doc in docs {
                let (result, delta) = process_doc(&doc, semantic_engine.clone(), weights, pid)?;
                results.push(result);
                rss_deltas.push(delta);
                sys.refresh_all();
                peak_rss_mb = peak_rss_mb.max(rss_mb(&mut sys, pid));
            }
        }
        Mode::SmallParallel => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(cli.workers)
                .build()
                .context("failed to build rayon pool")?;

            let out = pool.install(|| {
                docs.par_iter()
                    .map(|doc| process_doc(doc, semantic_engine.clone(), weights, pid))
                    .collect::<Vec<_>>()
            });

            for item in out {
                let (result, delta) = item?;
                results.push(result);
                rss_deltas.push(delta);
                sys.refresh_all();
                peak_rss_mb = peak_rss_mb.max(rss_mb(&mut sys, pid));
            }
        }
    }

    let total_batch_ms = batch_t0.elapsed().as_secs_f64() * 1000.0;

    results.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
    let processing_values = results.iter().map(|r| r.timing.processing_ms).collect::<Vec<_>>();
    let model_values = results
        .iter()
        .map(|r| r.timing.model_inference_ms)
        .collect::<Vec<_>>();

    let benchmark = BenchmarkResult {
        mode: match cli.mode {
            Mode::Sequential => "sequential".to_string(),
            Mode::SmallParallel => "small_parallel".to_string(),
        },
        docs_count: results.len(),
        rss_loaded_mb,
        peak_rss_mb,
        avg_rss_delta_per_doc_mb: mean(&rss_deltas),
        avg_processing_ms_per_doc: mean(&processing_values),
        p95_processing_ms_per_doc: p95(&processing_values),
        avg_model_inference_ms_per_doc: mean(&model_values),
        total_batch_ms,
    };

    let results_json = serde_json::to_string_pretty(&results)?;
    let benchmark_json = serde_json::to_string_pretty(&benchmark)?;

    let results_path = cli.output_dir.join("results_rust.json");
    let benchmark_path = cli.output_dir.join("benchmark_rust.json");

    fs::write(&results_path, results_json)?;
    fs::write(&benchmark_path, benchmark_json)?;

    println!("semantic_backend={}", semantic_engine.backend_name());
    println!("saved {}", results_path.display());
    println!("saved {}", benchmark_path.display());

    Ok(())
}

fn process_doc(
    doc: &LoadedDoc,
    semantic_engine: Arc<dyn SemanticEngine>,
    weights: ScoreWeights,
    pid: Pid,
) -> Result<(DocumentResult, f64)> {
    let mut sys = System::new_all();
    sys.refresh_all();
    let rss_before = rss_mb(&mut sys, pid);

    let t0 = Instant::now();
    let parsed = ParsedDoc::from_text(doc.text.clone());

    let coverage = coverage_features(&parsed.text, &parsed.lines, &parsed.tokens);
    let structure = structure_features(&parsed.text, &parsed.lines);
    let noise = noise_features(&parsed.text, &parsed.lines, &parsed.tokens);
    let (info_density, model_inference_ms) = semantic_engine.compute(&parsed.text, &parsed.tokens);

    let coverage_score_v = coverage_score(&coverage);
    let structure_score_v = structure_score(&structure);
    let noise_score_v = noise_score(&noise);
    let info_density_score_v = info_density_score(&info_density);

    let dqi_total_score = dqi_total(
        coverage_score_v,
        structure_score_v,
        info_density_score_v,
        noise_score_v,
        weights,
    );

    let processing_ms = t0.elapsed().as_secs_f64() * 1000.0;

    sys.refresh_all();
    let rss_after = rss_mb(&mut sys, pid);
    let rss_delta_mb = (rss_after - rss_before).max(0.0);

    let mut features = BTreeMap::<String, Value>::new();
    features.insert("coverage".to_string(), serde_json::to_value(coverage)?);
    features.insert("structure".to_string(), serde_json::to_value(structure)?);
    features.insert("noise".to_string(), serde_json::to_value(noise)?);
    features.insert("info_density".to_string(), serde_json::to_value(info_density)?);

    let result = DocumentResult {
        doc_id: doc
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        path: doc.path.display().to_string(),
        features,
        scores: Scores {
            coverage_score: coverage_score_v,
            structure_score: structure_score_v,
            noise_score: noise_score_v,
            info_density_score: info_density_score_v,
            perplexity_score: None,
            dqi_total: dqi_total_score,
        },
        timing: Timing {
            processing_ms,
            model_inference_ms,
            rss_delta_mb,
        },
    };

    Ok((result, rss_delta_mb))
}

fn collect_docs(input_dir: &Path) -> Result<Vec<LoadedDoc>> {
    let mut docs = Vec::new();
    for entry in fs::read_dir(input_dir)
        .with_context(|| format!("failed to read {}", input_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map(|e| e == "md").unwrap_or(false) {
            let text = fs::read_to_string(&path)
                .with_context(|| format!("failed to read markdown {}", path.display()))?;
            docs.push(LoadedDoc { path, text });
        }
    }
    docs.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(docs)
}

fn parse_weights(input: &str) -> Result<ScoreWeights> {
    let parts = input
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse --weights")?;
    if parts.len() != 4 {
        anyhow::bail!("--weights must contain exactly 4 comma-separated values")
    }
    Ok(ScoreWeights {
        coverage: parts[0],
        structure: parts[1],
        info_density: parts[2],
        noise: parts[3],
    })
}

fn rss_mb(sys: &mut System, pid: Pid) -> f64 {
    sys.refresh_processes();
    sys.process(pid)
        .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn p95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
    let idx = idx.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}
