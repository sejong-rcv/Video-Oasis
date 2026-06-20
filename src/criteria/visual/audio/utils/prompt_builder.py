def format_transcript_with_timestamps(stt_data):
    """
    Converts STT JSON data into timestamped text format.
    Example:
    [0.0 - 4.5] Hello, welcome to the video.
    [4.5 - 10.2] Today we are going to learn about...
    """
    if not stt_data or "timestamps" not in stt_data:
        return stt_data.get("transcript", "") if stt_data else "No transcript available."
    
    formatted_lines = []
    for chunk in stt_data["timestamps"]:
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()

        if start is None: start = 0.0
        if end is None: end = 0.0
        
        formatted_lines.append(f"[{start:.1f} - {end:.1f}] {text}")
    
    return "\n".join(formatted_lines)

def build_timestamp_prompt(stt_data, question, options):
    transcript_context = format_transcript_with_timestamps(stt_data)
    options_str = "\n".join(options)
    
    prompt = f"""You are an AI assistant tasked with answering questions based STRICTLY on the provided video transcript.

[INSTRUCTIONS]
1. Read the transcript with timestamps carefully.
2. Select the letter of the correct option from the provided choices (e.g., A, B, C, etc.).
3. **CRITICAL**: You MUST cite the specific timestamp range (e.g., [12.5 - 15.0]) from the transcript that supports your answer.
4. Do NOT use external knowledge.

[Transcript]
{transcript_context}

[Question]
{question}

[Options]
{options_str}

[Format]
Answer: (Option Letter)
Evidence: (Quote the text and timestamp that supports your answer)
Reasoning: (Brief explanation)"""
    
    return prompt.strip()