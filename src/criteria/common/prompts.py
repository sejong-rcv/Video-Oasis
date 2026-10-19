"""Prompt builders aligned with the Video-Oasis diagnostic protocol."""

from __future__ import annotations

from .annotations import Annotation


AUDIO_PROMPT_VERSION = "timestamp_evidence_v1"


def option_candidates(num_options: int) -> str:
    letters = [chr(ord("A") + i) for i in range(num_options)]
    if len(letters) == 1:
        return f"({letters[0]})"
    return f"({', '.join(letters[:-1])}, or {letters[-1]})"


def format_options(ann: Annotation) -> str:
    options = ann.get("options")
    if not options:
        raise ValueError("The sample has no answer options.")
    return "\n".join(options)


def build_visual_dependency_prompt(
    ann: Annotation,
    context_label: str,
    context: str = "",
) -> str:
    candidates = option_candidates(len(ann["options"]))
    context_block = context if context else "Empty"
    return (
        "You are a helpful assistant. Select the best answer to the following "
        "multiple-choice question based on the provided context and options.\n\n"
        f"Question:\n{ann['question']}\n\n"
        f"Context ({context_label}):\n{context_block}\n\n"
        f"Options:\n{format_options(ann)}\n\n"
        f"Respond with only the letter {candidates} of the correct option. "
        "Put your final answer in \\boxed{}."
    )


def build_audio_prompt(ann: Annotation, transcript: str) -> str:
    options = format_options(ann)
    last_letter = chr(ord("A") + len(ann["options"]) - 1)
    return f"""You are an AI assistant tasked with answering questions based STRICTLY on the provided video transcript.

[INSTRUCTIONS]
1. Read the transcript with timestamps carefully.
2. Select the correct option (A to {last_letter}).
3. **CRITICAL**: You MUST cite the specific timestamp range (e.g., [12.5 - 15.0]) from the transcript that supports your answer.
4. **UNCERTAINTY**: If the provided transcript does not contain enough information to answer the question, acknowledge the uncertainty in the 'Reasoning' section, but still select the most plausible option based on context.
5. **OUTPUT FORMAT**: For the 'Answer:' field, write ONLY the option letter (e.g., 'Answer: A'). Do not add any other text or parentheses.
[Transcript]
{transcript}

[Question]
{ann['question']}

[Options]
{options}

[Format]
Answer: (Option Letter)
Evidence: (Quote the text and timestamp. If none, write "None")
Reasoning: (Explain why you chose this option. If you are guessing, explicitly state "Information missing, guessing based on context.")"""


def build_mllm_video_prompt(ann: Annotation, visual_word: str = "video") -> str:
    candidates = option_candidates(len(ann["options"]))
    return (
        "You are a helpful assistant. Select the best answer to the following "
        f"multiple-choice question based on the {visual_word}, question, and options.\n\n"
        f"Question:\n{ann['question']}\n\n"
        f"Options:\n{format_options(ann)}\n\n"
        f"Respond with only the letter {candidates} of the correct option. "
        "Put your final answer in \\boxed{}."
    )
