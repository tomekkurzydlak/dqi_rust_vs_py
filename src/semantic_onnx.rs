use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;

pub type FeatureMap = BTreeMap<String, Value>;

static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was",
        "were", "be", "with", "by", "at", "as", "that", "this", "it", "from", "can", "will",
    ]
    .into_iter()
    .collect()
});

pub trait SemanticEngine: Send + Sync {
    fn compute(&self, text: &str, tokens: &[String]) -> (FeatureMap, f64);
    fn backend_name(&self) -> &'static str;
}

pub struct FallbackSemanticEngine;

impl SemanticEngine for FallbackSemanticEngine {
    fn compute(&self, text: &str, tokens: &[String]) -> (FeatureMap, f64) {
        let token_count = tokens.len();
        let stopword_count = tokens
            .iter()
            .filter(|t| STOPWORDS.contains(t.to_lowercase().as_str()))
            .count();
        let unique_token_ratio = tokens
            .iter()
            .map(|t| t.to_lowercase())
            .collect::<HashSet<_>>()
            .len() as f64
            / token_count.max(1) as f64;
        let lexical_density_proxy = tokens
            .iter()
            .filter(|t| !STOPWORDS.contains(t.to_lowercase().as_str()))
            .count() as f64
            / token_count.max(1) as f64;

        let entity_count = text
            .split_whitespace()
            .filter(|w| {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
                clean.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && clean.chars().skip(1).any(|c| c.is_lowercase())
                    && clean.len() > 2
            })
            .count();

        let entity_density = entity_count as f64 / token_count.max(1) as f64;
        (
            BTreeMap::from([
                ("token_count".to_string(), json!(token_count)),
                (
                    "stopword_ratio".to_string(),
                    json!(stopword_count as f64 / token_count.max(1) as f64),
                ),
                ("entity_count".to_string(), json!(entity_count)),
                ("entity_density".to_string(), json!(entity_density)),
                ("unique_token_ratio".to_string(), json!(unique_token_ratio)),
                (
                    "lexical_density_proxy".to_string(),
                    json!(lexical_density_proxy),
                ),
                ("semantic_backend".to_string(), json!("fallback_no_onnx")),
            ]),
            0.0,
        )
    }

    fn backend_name(&self) -> &'static str {
        "fallback_no_onnx"
    }
}

pub struct OnnxNerSemanticEngine {
    tokenizer: Tokenizer,
    sessions: Arc<Vec<Mutex<Session>>>,
    next_session: AtomicUsize,
    max_len: usize,
}

impl OnnxNerSemanticEngine {
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        max_len: usize,
        intra_threads: usize,
        inter_threads: usize,
        session_pool_size: usize,
    ) -> Result<Self> {
        if session_pool_size == 0 {
            anyhow::bail!("session_pool_size must be >= 1");
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .with_context(|| format!("failed to load tokenizer from {}", tokenizer_path.display()))?;

        let mut sessions = Vec::with_capacity(session_pool_size);
        for _ in 0..session_pool_size {
            let mut builder = Session::builder()?
                .with_intra_threads(intra_threads)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_inter_threads(inter_threads)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::All)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;

            let session = builder
                .commit_from_file(model_path)
                .with_context(|| format!("failed to load ONNX from {}", model_path.display()))?;
            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            tokenizer,
            sessions: Arc::new(sessions),
            next_session: AtomicUsize::new(0),
            max_len,
        })
    }

    fn infer_entity_count(&self, text: &str) -> Result<(usize, f64)> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut input_ids = encoded
            .get_ids()
            .iter()
            .map(|v| *v as i64)
            .collect::<Vec<_>>();
        let mut attention_mask = encoded
            .get_attention_mask()
            .iter()
            .map(|v| *v as i64)
            .collect::<Vec<_>>();
        let mut token_type_ids = encoded
            .get_type_ids()
            .iter()
            .map(|v| *v as i64)
            .collect::<Vec<_>>();

        // Keep inference bounded and stable for benchmark repeatability.
        if input_ids.len() > self.max_len {
            input_ids.truncate(self.max_len);
            attention_mask.truncate(self.max_len);
            token_type_ids.truncate(self.max_len);
        }

        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Ok((0, 0.0));
        }

        let input_ids_tensor = Tensor::<i64>::from_array(([1i64, seq_len as i64], input_ids))?;
        let attention_tensor = Tensor::<i64>::from_array(([1i64, seq_len as i64], attention_mask))?;
        let token_type_tensor =
            Tensor::<i64>::from_array(([1i64, seq_len as i64], token_type_ids))?;

        let t0 = Instant::now();
        let session_idx = self.next_session.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        let mut session = self.sessions[session_idx]
            .lock()
            .expect("onnx session mutex poisoned");
        let input_count = session.inputs().len();

        let outputs = match input_count {
            2 => session.run(ort::inputs![input_ids_tensor, attention_tensor])?,
            _ => session.run(ort::inputs![
                input_ids_tensor,
                attention_tensor,
                token_type_tensor
            ])?,
        };
        let model_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let logits = outputs[0].try_extract_array::<f32>()?;
        let shape = logits.shape();
        if shape.len() < 3 {
            return Ok((0, model_ms));
        }

        let mut entity_tokens = 0usize;
        for i in 0..shape[1] {
            let mut max_idx = 0usize;
            let mut max_val = f32::MIN;
            for label in 0..shape[2] {
                let v = logits[[0, i, label]];
                if v > max_val {
                    max_val = v;
                    max_idx = label;
                }
            }
            if max_idx != 0 {
                entity_tokens += 1;
            }
        }

        Ok((entity_tokens, model_ms))
    }
}

impl SemanticEngine for OnnxNerSemanticEngine {
    fn compute(&self, text: &str, tokens: &[String]) -> (FeatureMap, f64) {
        let token_count = tokens.len();
        let stopword_count = tokens
            .iter()
            .filter(|t| STOPWORDS.contains(t.to_lowercase().as_str()))
            .count();
        let unique_token_ratio = tokens
            .iter()
            .map(|t| t.to_lowercase())
            .collect::<HashSet<_>>()
            .len() as f64
            / token_count.max(1) as f64;
        let lexical_density_proxy = tokens
            .iter()
            .filter(|t| !STOPWORDS.contains(t.to_lowercase().as_str()))
            .count() as f64
            / token_count.max(1) as f64;

        let (entity_count, model_ms) = match self.infer_entity_count(text) {
            Ok(v) => v,
            Err(_) => (0usize, 0.0),
        };
        let entity_density = entity_count as f64 / token_count.max(1) as f64;

        (
            BTreeMap::from([
                ("token_count".to_string(), json!(token_count)),
                (
                    "stopword_ratio".to_string(),
                    json!(stopword_count as f64 / token_count.max(1) as f64),
                ),
                ("entity_count".to_string(), json!(entity_count)),
                ("entity_density".to_string(), json!(entity_density)),
                ("unique_token_ratio".to_string(), json!(unique_token_ratio)),
                (
                    "lexical_density_proxy".to_string(),
                    json!(lexical_density_proxy),
                ),
                ("semantic_backend".to_string(), json!("onnx_token_classification")),
            ]),
            model_ms,
        )
    }

    fn backend_name(&self) -> &'static str {
        "onnx_token_classification"
    }
}
