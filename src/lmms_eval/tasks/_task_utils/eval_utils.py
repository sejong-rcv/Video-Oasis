import re

from loguru import logger as eval_logger


ANSWER_LETTERS = tuple("ABCDEFGHIJKLMN")
ERROR_SENTINEL_RE = re.compile(r"^error::[a-z0-9_]+$", re.IGNORECASE)
BOXED_ANSWER_RE = re.compile(
    r"\\?boxed\s*\{\s*(?:\\(?:text|textbf|mathrm|mathbf)\s*\{\s*)?"
    r"[\(\[]?\s*([A-N])\b",
    re.IGNORECASE,
)
TAGGED_ANSWER_RE = re.compile(r"<answer>\s*([A-N])\s*</answer>", re.IGNORECASE)
EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:(?:final|correct|best)\s+(?:answer|option)|answer)\s*"
    r"(?:is\s*|[:=]\s*)(?:\\?boxed\s*\{\s*)?[\(\[]?\s*([A-N])\b",
    re.IGNORECASE,
)
CORRECT_OPTION_RE = re.compile(
    r"\b(?:option|choice)\s+[\(\[]?\s*([A-N])\s*[\)\]]?\s+"
    r"(?:is\s+)?(?:correct|best)\b",
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


def extract_answer_letter(text, num_options=None):
    """Extract a final multiple-choice answer without scanning arbitrary prose."""
    if isinstance(text, dict):
        text = text.get("content", "")
    text = str(text).strip()

    if ERROR_SENTINEL_RE.fullmatch(text):
        return text.lower()

    if num_options is None:
        allowed = ANSWER_LETTERS
    else:
        allowed = ANSWER_LETTERS[: max(0, int(num_options))]

    for pattern in (
        TAGGED_ANSWER_RE,
        BOXED_ANSWER_RE,
        EXPLICIT_ANSWER_RE,
        CORRECT_OPTION_RE,
        LEADING_ANSWER_RE,
        STANDALONE_ANSWER_RE,
    ):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        letter = matches[-1].group(1).upper()
        return letter if letter in allowed else ""
    return ""


def extract_after_think(text):
    """
    Extracts the content after the last </think> tag in the given text.

    Args:
        text (str): The text containing </think> tags

    Returns:
        str: The content after the last </think> tag, or the original text if no </think> tag is found
    """
    # Find the last occurrence of </think>
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        think_content = text[:last_think_end].strip()
        answer_content = text[last_think_end + len("</think>") :].strip()

        think_matches = re.findall(
            r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", think_content
        )
        answer_matches = re.findall(
            r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", answer_content
        )

        if think_matches and not answer_matches:
            eval_logger.info(
                f"\nBoxed not found in answer but found in think, use think content: {text}"
            )
            return think_matches[-1]

        return answer_content
    else:
        eval_logger.info(f"\nCould not find </think>, use full text: {text}")
        return text


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
