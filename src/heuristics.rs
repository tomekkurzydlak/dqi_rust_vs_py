use crate::parser::{BROKEN_LIST_RE, CONTROL_RE, HEADING_RE, LIST_RE};
use once_cell::sync::Lazy;
use regex::Regex;
use crate::scoring::clamp_0_100;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

pub type FeatureMap = BTreeMap<String, Value>;
static PARAGRAPH_SPLIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n\s*\n").unwrap());
static GLUE_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z]{6,}[0-9]{3,}|[a-z]{3,}[A-Z]{3,}").unwrap());

pub fn coverage_features(text: &str, lines: &[String], tokens: &[String]) -> FeatureMap {
    let headings = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| HEADING_RE.is_match(l))
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();

    let nonempty_line_count = lines.iter().filter(|l| !l.trim().is_empty()).count();

    let paragraphs = PARAGRAPH_SPLIT_RE
        .split(text)
        .filter(|blk| {
            let t = blk.trim();
            !t.is_empty() && !HEADING_RE.is_match(t)
        })
        .count();

    let mut empty_section_count = 0usize;
    for (idx, start) in headings.iter().enumerate() {
        let mut end = lines.len();
        if let Some(next_start) = headings.get(idx + 1) {
            end = *next_start;
        }
        let body = lines[start + 1..end].join("\n");
        if body.trim().is_empty() {
            empty_section_count += 1;
        }
    }

    let section_count = headings.len();
    let heading_count = section_count;
    let content_line_ratio = nonempty_line_count as f64 / lines.len().max(1) as f64;
    let empty_section_ratio = empty_section_count as f64 / section_count.max(1) as f64;

    BTreeMap::from([
        ("char_count".to_string(), json!(text.chars().count())),
        ("word_count".to_string(), json!(tokens.len())),
        ("nonempty_line_count".to_string(), json!(nonempty_line_count)),
        ("paragraph_count".to_string(), json!(paragraphs)),
        ("heading_count".to_string(), json!(heading_count)),
        ("section_count".to_string(), json!(section_count)),
        ("empty_section_count".to_string(), json!(empty_section_count)),
        ("content_line_ratio".to_string(), json!(content_line_ratio)),
        ("empty_section_ratio".to_string(), json!(empty_section_ratio)),
    ])
}

pub fn coverage_score(features: &FeatureMap) -> f64 {
    let mut score = 100.0;
    let word_count = as_f64(features, "word_count");
    let paragraph_count = as_f64(features, "paragraph_count");
    let content_line_ratio = as_f64(features, "content_line_ratio");
    let empty_section_ratio = as_f64(features, "empty_section_ratio");

    if word_count < 80.0 {
        score -= (80.0 - word_count) * 0.45;
    }
    if paragraph_count < 2.0 {
        score -= (2.0 - paragraph_count) * 10.0;
    }
    if content_line_ratio < 0.55 {
        score -= (0.55 - content_line_ratio) * 85.0;
    }
    score -= empty_section_ratio * 50.0;

    clamp_0_100(score)
}

pub fn structure_features(text: &str, lines: &[String]) -> FeatureMap {
    let heading_lines = lines
        .iter()
        .filter_map(|l| HEADING_RE.captures(l))
        .collect::<Vec<_>>();

    let heading_count = heading_lines.len();
    let section_count = heading_count;
    let paragraph_count = PARAGRAPH_SPLIT_RE.split(text).filter(|b| !b.trim().is_empty()).count();
    let list_count = lines.iter().filter(|l| LIST_RE.is_match(l)).count();
    let table_count = lines
        .iter()
        .filter(|l| l.contains('|') && l.matches('|').count() >= 2)
        .count();

    let broken_heading_count = heading_lines
        .iter()
        .filter(|caps| caps.get(2).map(|m| m.as_str().trim()).unwrap_or("").is_empty())
        .count();
    let empty_heading_count = broken_heading_count;
    let broken_list_marker_count = lines.iter().filter(|l| BROKEN_LIST_RE.is_match(l)).count();

    let table_heavy_flag = table_count > usize::max(2, paragraph_count);
    let narrative_heavy_flag = paragraph_count > usize::max(3, table_count * 4);

    BTreeMap::from([
        ("heading_count".to_string(), json!(heading_count)),
        ("section_count".to_string(), json!(section_count)),
        ("paragraph_count".to_string(), json!(paragraph_count)),
        ("list_count".to_string(), json!(list_count)),
        ("table_count".to_string(), json!(table_count)),
        ("broken_heading_count".to_string(), json!(broken_heading_count)),
        ("empty_heading_count".to_string(), json!(empty_heading_count)),
        (
            "broken_list_marker_count".to_string(),
            json!(broken_list_marker_count),
        ),
        ("table_heavy_flag".to_string(), json!(table_heavy_flag)),
        ("narrative_heavy_flag".to_string(), json!(narrative_heavy_flag)),
    ])
}

pub fn structure_score(features: &FeatureMap) -> f64 {
    let mut score = 100.0;
    let heading_count = as_f64(features, "heading_count");
    let broken_heading_count = as_f64(features, "broken_heading_count");
    let empty_heading_count = as_f64(features, "empty_heading_count");
    let broken_list_marker_count = as_f64(features, "broken_list_marker_count");

    if heading_count == 0.0 {
        score -= 35.0;
    }
    score -= broken_heading_count * 10.0;
    score -= empty_heading_count * 8.0;
    score -= broken_list_marker_count * 7.0;

    let table_heavy = features
        .get("table_heavy_flag")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let narrative_heavy = features
        .get("narrative_heavy_flag")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if table_heavy && narrative_heavy {
        score -= 8.0;
    }

    clamp_0_100(score)
}

pub fn noise_features(text: &str, lines: &[String], _tokens: &[String]) -> FeatureMap {
    let tokens = text
        .split_whitespace()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let replacement_char_count = text.matches('\u{fffd}').count() + text.matches('�').count();
    let control_char_count = CONTROL_RE.find_iter(text).count();

    let weird_unicode_count = text
        .chars()
        .filter(|ch| *ch as u32 > 127 && !ch.is_alphabetic())
        .count();
    let weird_unicode_ratio = weird_unicode_count as f64 / text.chars().count().max(1) as f64;

    let long_token_ratio = tokens.iter().filter(|t| t.len() >= 30).count() as f64 / tokens.len().max(1) as f64;

    let punct_count = text
        .chars()
        .filter(|ch| "!?.,;:-_+=*/\\|[]{}()<>~`\"'@#$%^&".contains(*ch))
        .count();
    let digit_count = text.chars().filter(|ch| ch.is_ascii_digit()).count();
    let punctuation_ratio = punct_count as f64 / text.chars().count().max(1) as f64;
    let digit_ratio = digit_count as f64 / text.chars().count().max(1) as f64;

    let garbage_token_ratio = tokens
        .iter()
        .filter(|t| {
            t.len() >= 8
                && t.chars().filter(|c| !c.is_alphanumeric()).count() as f64 / t.len() as f64
                    > 0.35
        })
        .count() as f64
        / tokens.len().max(1) as f64;

    let glue_like_token_ratio = tokens
        .iter()
        .filter(|t| GLUE_TOKEN_RE.is_match(t))
        .count() as f64
        / tokens.len().max(1) as f64;

    let mut counter = HashMap::<String, usize>::new();
    for l in lines.iter().map(|s| s.trim()).filter(|l| !l.is_empty()) {
        *counter.entry(l.to_string()).or_default() += 1;
    }
    let repeated_lines = counter.values().filter(|v| **v > 1).map(|v| v - 1).sum::<usize>();
    let repeated_line_ratio = repeated_lines as f64 / counter.values().sum::<usize>().max(1) as f64;

    BTreeMap::from([
        ("replacement_char_count".to_string(), json!(replacement_char_count)),
        ("control_char_count".to_string(), json!(control_char_count)),
        ("weird_unicode_ratio".to_string(), json!(weird_unicode_ratio)),
        ("long_token_ratio".to_string(), json!(long_token_ratio)),
        ("punctuation_ratio".to_string(), json!(punctuation_ratio)),
        ("digit_ratio".to_string(), json!(digit_ratio)),
        ("garbage_token_ratio".to_string(), json!(garbage_token_ratio)),
        ("glue_like_token_ratio".to_string(), json!(glue_like_token_ratio)),
        ("repeated_line_ratio".to_string(), json!(repeated_line_ratio)),
    ])
}

pub fn noise_score(features: &FeatureMap) -> f64 {
    let mut score = 100.0;
    score -= as_f64(features, "replacement_char_count") * 2.0;
    score -= as_f64(features, "control_char_count") * 1.5;
    score -= as_f64(features, "weird_unicode_ratio") * 300.0;
    score -= as_f64(features, "long_token_ratio") * 120.0;
    score -= as_f64(features, "punctuation_ratio") * 80.0;
    score -= as_f64(features, "digit_ratio") * 40.0;
    score -= as_f64(features, "garbage_token_ratio") * 140.0;
    score -= as_f64(features, "glue_like_token_ratio") * 120.0;
    score -= as_f64(features, "repeated_line_ratio") * 110.0;
    clamp_0_100(score)
}

pub fn info_density_score(features: &FeatureMap) -> f64 {
    let token_count = as_f64(features, "token_count");
    let stopword_ratio = as_f64(features, "stopword_ratio");
    let entity_density = as_f64(features, "entity_density");
    let unique_token_ratio = as_f64(features, "unique_token_ratio");
    let lexical_density_proxy = as_f64(features, "lexical_density_proxy");

    let mut score = 0.0;
    score += (token_count / 20.0).min(30.0);
    score += (unique_token_ratio * 30.0).min(20.0);
    score += (lexical_density_proxy * 25.0).min(20.0);
    score += (entity_density * 250.0).min(20.0);
    score += (10.0 - (stopword_ratio - 0.45).abs() * 25.0).max(0.0);
    clamp_0_100(score)
}

fn as_f64(map: &FeatureMap, key: &str) -> f64 {
    map.get(key).and_then(|v| v.as_f64()).unwrap_or_else(|| {
        map.get(key)
            .and_then(|v| v.as_i64())
            .map(|v| v as f64)
            .unwrap_or(0.0)
    })
}
