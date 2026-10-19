import re

from loguru import logger as eval_logger


ANSWER_LETTERS = tuple("ABCDEFGHIJKLMN")
ERROR_SENTINEL_RE = re.compile(r"^error::[a-z0-9_]+$", re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(r"\bfinal\s+answer\s*(?:is\b|[:=])\s*", re.IGNORECASE)
ANSWER_PREFIX_RE = re.compile(r"^(?:the\s+)?answer\s*(?:is\b|[:=])\s*", re.IGNORECASE)
ANSWER_PATTERNS = (
    r"([A-N])\.?",
    r"\(\s*([A-N])\s*\)\.?",
    r"\\?boxed\s*\{\s*([A-N])\s*\}",
    r"<answer>\s*([A-N])\s*</answer>",
)


def extract_answer_letter(text, num_options=None):
    """Accept only a single formatted answer in the final response."""
    if isinstance(text, dict):
        text = text.get("content", "")
    text = extract_after_think(str(text).strip())

    if ERROR_SENTINEL_RE.fullmatch(text):
        return text.lower()

    if num_options is None:
        allowed = ANSWER_LETTERS
    else:
        allowed = ANSWER_LETTERS[: max(0, int(num_options))]

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


def extract_after_think(text):
    """Keep only the final response; an unfinished think block has no final answer."""
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    return "" if re.search(r"<think>", text, re.IGNORECASE) else text


def extract_final_boxed_content(text, strict=False):
    """
    Extracts the content of the final \\boxed{} command in the given text.

    Args:
        text (str): The text containing \\boxed{} commands

    Returns:
        str or None: The content of the final \\boxed{} command, or None if no \\boxed{} command is found
    """
    # extract the content after the last </think> tag
    # text = extract_after_think(text)

    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    # Return the last match if any matches were found
    if boxed_matches:
        return boxed_matches[-1]
    else:
        eval_logger.info(f"no boxed found in {text}")
        if strict:
            return ""
        else:
            return text
