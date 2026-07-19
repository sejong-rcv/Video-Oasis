"""Prediction parsing helpers."""

from __future__ import annotations

import re

from .constants import ANSWER_LETTERS


BOXED_ANSWER_RE = re.compile(
    r"\\?boxed\s*\{\s*(?:\\(?:text|textbf|mathrm|mathbf)\s*\{\s*)?"
    r"[\(\[]?\s*([A-N])\b",
    re.IGNORECASE,
)
TAGGED_ANSWER_RE = re.compile(r"<answer>\s*([A-N])\s*</answer>", re.IGNORECASE)
EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:final\s+answer|answer)\s*(?:is\s*|[:=]\s*)"
    r"(?:\\?boxed\s*\{\s*)?[\(\[]?\s*([A-N])\b",
    re.IGNORECASE,
)
LEADING_ANSWER_RE = re.compile(
    r"^\s*[`*_]*[\(\[]?\s*([A-N])\s*(?:[\)\].:]|-)\s+\S",
    re.IGNORECASE,
)
STANDALONE_ANSWER_RE = re.compile(
    r"^\s*[`*_]*[\(\[]?\s*([A-N])\s*[\)\].]?[`*_]*\s*$",
    re.IGNORECASE,
)


def extract_answer_letter(text: object, num_options: int | None = None) -> str:
    if isinstance(text, dict):
        text = text.get("content", "")
    text = str(text).strip()

    allowed = ANSWER_LETTERS[:num_options] if num_options else ANSWER_LETTERS
    for pattern in (
        TAGGED_ANSWER_RE,
        BOXED_ANSWER_RE,
        EXPLICIT_ANSWER_RE,
        LEADING_ANSWER_RE,
        STANDALONE_ANSWER_RE,
    ):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        letter = matches[-1].group(1).upper()
        return letter if letter in allowed else ""
    return ""
