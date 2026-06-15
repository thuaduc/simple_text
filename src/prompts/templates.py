"""Default prompt for sentence-level biomedical text simplification."""


DEFAULT_SYSTEM_PROMPT = (
    "You simplify biomedical sentences. Be conservative. "
    "Keep the sentence structure when possible. Only change what "
    "is necessary to make it understandable to a general audience."
)


def create_default_prompt(input_sentence: str, tokenizer=None) -> str:
    """Create the single default simplification prompt."""
    user_prompt = (
        "Simplify this biomedical sentence for a lay reader.\n"
        "Rules:\n"
        "- Replace medical jargon with plain words\n"
        "- Remove statistical details (CI, p-values, RR, OR)\n"
        "- Keep all key facts and numbers\n"
        "- If already simple, return it unchanged\n"
        "- Do NOT add new information\n"
        "- Output ONLY the simplified sentence\n\n"
        f"Sentence: {input_sentence}\n"
        "Simplified:"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"

