use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Serialize, Clone)]
pub struct Timing {
    pub processing_ms: f64,
    pub model_inference_ms: f64,
    pub rss_delta_mb: f64,
}

#[derive(Serialize, Clone)]
pub struct Scores {
    pub coverage_score: f64,
    pub structure_score: f64,
    pub noise_score: f64,
    pub info_density_score: f64,
    pub perplexity_score: Option<f64>,
    pub dqi_total: f64,
}

#[derive(Serialize, Clone)]
pub struct DocumentResult {
    pub doc_id: String,
    pub path: String,
    pub features: BTreeMap<String, Value>,
    pub scores: Scores,
    pub timing: Timing,
}

#[derive(Serialize)]
pub struct BenchmarkResult {
    pub mode: String,
    pub docs_count: usize,
    pub rss_loaded_mb: f64,
    pub peak_rss_mb: f64,
    pub avg_rss_delta_per_doc_mb: f64,
    pub avg_processing_ms_per_doc: f64,
    pub p95_processing_ms_per_doc: f64,
    pub avg_model_inference_ms_per_doc: f64,
    pub total_batch_ms: f64,
}
