"""Prediction parsing helpers."""

from __future__ import annotations

import re

from .constants import ANSWER_LETTERS


FINAL_ANSWER_RE = re.compile(r"\bfinal\s+answer\s*(?:is\b|[:=])\s*", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(r"^(?:the\s+)?answer\s*(?:is\b|[:=])\s*", re.IGNORECASE)
ANSWER_PATTERNS = (
    r"([A-N])\.?",
    r"\(\s*([A-N])\s*\)\.?",
    r"\\?boxed\s*\{\s*([A-N])\s*\}",
    r"<answer>\s*([A-N])\s*</answer>",
)


def extract_answer_letter(text: object, num_options: int | None = None) -> str:
    """Accept only a single formatted answer in the final response."""
    if isinstance(text, dict):
        text = text.get("content", "")
    text = str(text).strip()
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    if re.search(r"<think>", text, re.IGNORECASE):
        return ""

    allowed = ANSWER_LETTERS if num_options is None else ANSWER_LETTERS[:max(0, int(num_options))]
    final_answers = list(FINAL_ANSWER_RE.finditer(text))
    if len(final_answers) > 1:
        return ""
    if final_answers:
        text = text[final_answers[0].end():].strip()
    else:
        text = ANSWER_PREFIX_RE.sub("", text, count=1)
    if len(text) >= 2 and text[0] == text[-1] == "$":
        text = text[1:-1].strip()

    for pattern in ANSWER_PATTERNS:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            return letter if letter in allowed else ""
    return ""
