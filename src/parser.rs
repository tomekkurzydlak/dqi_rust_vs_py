use once_cell::sync::Lazy;
use regex::Regex;

pub static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9_]+").unwrap());
pub static HEADING_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(#{1,6})\s*(.*)$").unwrap());
pub static LIST_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*([-*+]\s+|\d+[.)]\s+)").unwrap());
pub static BROKEN_LIST_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*[*+-](\S)").unwrap());
pub static CONTROL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").unwrap());

#[derive(Clone)]
pub struct ParsedDoc {
    pub text: String,
    pub lines: Vec<String>,
    pub tokens: Vec<String>,
}

impl ParsedDoc {
    pub fn from_text(text: String) -> Self {
        let lines = text.lines().map(ToString::to_string).collect::<Vec<_>>();
        let tokens = WORD_RE
            .find_iter(&text)
            .map(|m| m.as_str().to_string())
            .collect::<Vec<_>>();
        Self { text, lines, tokens }
    }

}
