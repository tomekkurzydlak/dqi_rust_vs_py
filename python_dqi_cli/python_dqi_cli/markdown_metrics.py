from __future__ import annotations

import re
from collections import Counter

from .scoring import clamp_0_100

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
HEADING_RE = re.compile(r"^(#{1,6})\s*(.*)$")
LIST_RE = re.compile(r"^\s*([-*+]\s+|\d+[.)]\s+)")
BROKEN_LIST_RE = re.compile(r"^\s*[*+-](\S)")
TABLE_ROW_RE = re.compile(r"\|")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
LONG_TOKEN_RE = re.compile(r"\S{30,}")
GLUE_TOKEN_RE = re.compile(r"[A-Za-z]{6,}[0-9]{3,}|[a-z]{3,}[A-Z]{3,}")


STOPWORDS = {
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


def split_lines(text: str) -> list[str]:
    return text.splitlines()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def compute_coverage_features(text: str) -> dict[str, float | int]:
    lines = split_lines(text)
    nonempty_lines = [ln for ln in lines if ln.strip()]
    headings = [ln for ln in lines if HEADING_RE.match(ln)]

    paragraph_count = len(
        [blk for blk in re.split(r"\n\s*\n", text) if blk.strip() and not HEADING_RE.match(blk.strip())]
    )
    section_count = len(headings)

    empty_section_count = 0
    for i, heading_line in enumerate(headings):
        try:
            start_idx = lines.index(heading_line)
        except ValueError:
            continue
        end_idx = len(lines)
        for j in range(start_idx + 1, len(lines)):
            if HEADING_RE.match(lines[j]):
                end_idx = j
                break
        section_body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        if not section_body:
            empty_section_count += 1

    char_count = len(text)
    word_count = len(tokenize(text))
    nonempty_line_count = len(nonempty_lines)
    heading_count = len(headings)
    content_line_ratio = nonempty_line_count / max(1, len(lines))
    empty_section_ratio = empty_section_count / max(1, section_count)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "nonempty_line_count": nonempty_line_count,
        "paragraph_count": paragraph_count,
        "heading_count": heading_count,
        "section_count": section_count,
        "empty_section_count": empty_section_count,
        "content_line_ratio": content_line_ratio,
        "empty_section_ratio": empty_section_ratio,
    }


def compute_coverage_score(features: dict[str, float | int]) -> float:
    word_count = float(features["word_count"])
    paragraph_count = float(features["paragraph_count"])
    content_line_ratio = float(features["content_line_ratio"])
    empty_section_ratio = float(features["empty_section_ratio"])

    score = 100.0
    if word_count < 80:
        score -= (80 - word_count) * 0.45
    if paragraph_count < 2:
        score -= (2 - paragraph_count) * 10.0
    if content_line_ratio < 0.55:
        score -= (0.55 - content_line_ratio) * 85.0
    score -= empty_section_ratio * 50.0

    return clamp_0_100(score)


def compute_structure_features(text: str) -> dict[str, float | int | bool]:
    lines = split_lines(text)
    heading_matches = [HEADING_RE.match(ln) for ln in lines]
    heading_matches = [m for m in heading_matches if m]

    heading_count = len(heading_matches)
    section_count = heading_count
    paragraph_count = len([blk for blk in re.split(r"\n\s*\n", text) if blk.strip()])
    list_count = sum(1 for ln in lines if LIST_RE.match(ln))
    table_count = sum(1 for ln in lines if TABLE_ROW_RE.search(ln) and ln.count("|") >= 2)

    broken_heading_count = sum(1 for m in heading_matches if m.group(2).strip() == "")
    empty_heading_count = broken_heading_count
    broken_list_marker_count = sum(1 for ln in lines if BROKEN_LIST_RE.search(ln))

    table_heavy_flag = table_count > max(2, paragraph_count)
    narrative_heavy_flag = paragraph_count > max(3, table_count * 4)

    return {
        "heading_count": heading_count,
        "section_count": section_count,
        "paragraph_count": paragraph_count,
        "list_count": list_count,
        "table_count": table_count,
        "broken_heading_count": broken_heading_count,
        "empty_heading_count": empty_heading_count,
        "broken_list_marker_count": broken_list_marker_count,
        "table_heavy_flag": table_heavy_flag,
        "narrative_heavy_flag": narrative_heavy_flag,
    }


def compute_structure_score(features: dict[str, float | int | bool]) -> float:
    heading_count = float(features["heading_count"])
    broken_heading_count = float(features["broken_heading_count"])
    empty_heading_count = float(features["empty_heading_count"])
    broken_list_marker_count = float(features["broken_list_marker_count"])

    score = 100.0
    if heading_count == 0:
        score -= 35.0

    score -= broken_heading_count * 10.0
    score -= empty_heading_count * 8.0
    score -= broken_list_marker_count * 7.0

    if bool(features["table_heavy_flag"]) and bool(features["narrative_heavy_flag"]):
        score -= 8.0

    return clamp_0_100(score)


def compute_noise_features(text: str) -> dict[str, float | int]:
    tokens = text.split()
    lines = [ln.strip() for ln in split_lines(text) if ln.strip()]

    replacement_char_count = text.count("\ufffd") + text.count("�")
    control_char_count = len(CONTROL_RE.findall(text))

    weird_unicode_count = sum(1 for ch in text if ord(ch) > 127 and not ch.isalpha())
    weird_unicode_ratio = weird_unicode_count / max(1, len(text))

    long_token_count = sum(1 for t in tokens if LONG_TOKEN_RE.fullmatch(t))
    long_token_ratio = long_token_count / max(1, len(tokens))

    punct_count = sum(1 for ch in text if ch in "!?.,;:-_+=*/\\|[]{}()<>~`\"'@#$%^&")
    digit_count = sum(1 for ch in text if ch.isdigit())
    punctuation_ratio = punct_count / max(1, len(text))
    digit_ratio = digit_count / max(1, len(text))

    garbage_token_count = sum(
        1
        for t in tokens
        if len(t) >= 8 and (sum(1 for c in t if not c.isalnum()) / max(1, len(t))) > 0.35
    )
    garbage_token_ratio = garbage_token_count / max(1, len(tokens))

    glue_like_token_count = sum(1 for t in tokens if GLUE_TOKEN_RE.search(t))
    glue_like_token_ratio = glue_like_token_count / max(1, len(tokens))

    line_counter = Counter(lines)
    repeated_lines = sum(c - 1 for c in line_counter.values() if c > 1)
    repeated_line_ratio = repeated_lines / max(1, len(lines))

    return {
        "replacement_char_count": replacement_char_count,
        "control_char_count": control_char_count,
        "weird_unicode_ratio": weird_unicode_ratio,
        "long_token_ratio": long_token_ratio,
        "punctuation_ratio": punctuation_ratio,
        "digit_ratio": digit_ratio,
        "garbage_token_ratio": garbage_token_ratio,
        "glue_like_token_ratio": glue_like_token_ratio,
        "repeated_line_ratio": repeated_line_ratio,
    }


def compute_noise_score(features: dict[str, float | int]) -> float:
    score = 100.0
    score -= float(features["replacement_char_count"]) * 2.0
    score -= float(features["control_char_count"]) * 1.5
    score -= float(features["weird_unicode_ratio"]) * 300.0
    score -= float(features["long_token_ratio"]) * 120.0
    score -= float(features["punctuation_ratio"]) * 80.0
    score -= float(features["digit_ratio"]) * 40.0
    score -= float(features["garbage_token_ratio"]) * 140.0
    score -= float(features["glue_like_token_ratio"]) * 120.0
    score -= float(features["repeated_line_ratio"]) * 110.0
    return clamp_0_100(score)


def lexical_density_proxy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    content = [t for t in tokens if t.lower() not in STOPWORDS]
    return len(content) / len(tokens)
