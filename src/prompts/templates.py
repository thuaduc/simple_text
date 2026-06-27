"""Prompt templates for sentence-level biomedical text simplification."""

from typing import List, Tuple, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You simplify biomedical sentences. Be conservative. "
    "Keep the sentence structure when possible. Only change what "
    "is necessary to make it understandable to a general audience."
)

RAG_POSTEDIT_SYSTEM_PROMPT = (
    "You are a careful editor. You are given an original biomedical sentence, a "
    "draft simplification of it, and plain-language definitions for technical "
    "terms that still appear in the draft. Improve the draft only where needed so "
    "a general reader understands it, without changing the meaning of the original."
)


def _chat_prompt_no_think(system: str, user: str, tokenizer) -> str:
    """Build a chat prompt with model-specific formatting and thinking disabled."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _definitions_block(definitions: Optional[List[Tuple[str, str]]]) -> str:
    """Render retrieved glossary definitions as a prompt block (empty if none)."""
    if not definitions:
        return ""
    block = "Relevant definitions:\n"
    for term, definition in definitions:
        block += f"- {term}: {definition}\n"
    return block + "\n"


def create_default_prompt(
    input_sentence: str,
    tokenizer=None,
    definitions: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Create the default zero-shot simplification prompt.

    If ``definitions`` is provided (RAG, before-generation mode), a glossary
    definitions block and an extra rule are injected into the prompt.
    """
    definitions_block = _definitions_block(definitions)
    definitions_rule = (
        "- Use the definitions below ONLY to simplify terms that appear in the sentence\n"
        if definitions
        else ""
    )

    user_prompt = (
        "Simplify this biomedical sentence for a lay reader.\n"
        "Rules:\n"
        "- Replace medical jargon with plain words\n"
        "- Remove statistical details (CI, p-values, RR, OR)\n"
        "- Keep all key facts and numbers\n"
        "- If already simple, return it unchanged\n"
        "- Do NOT add new information\n"
        f"{definitions_rule}"
        "- Output ONLY the simplified sentence\n\n"
        f"{definitions_block}"
        f"Sentence: {input_sentence}\n"
        "Simplified:"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(DEFAULT_SYSTEM_PROMPT, user_prompt, tokenizer)

    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_few_shot_prompt(
    input_sentence: str,
    examples: List[Tuple[str, str]],
    tokenizer=None,
    definitions: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Create a few-shot prompt with retrieved examples.

    If ``definitions`` is provided (RAG, before-generation mode), a glossary
    definitions block and an extra rule are injected into the prompt.

    Args:
        input_sentence: The sentence to simplify
        examples: List of (complex, simple) example pairs
        tokenizer: Optional tokenizer for chat template
        definitions: Optional list of (term, definition) tuples from glossary retrieval
    """
    examples_text = ""
    if examples:
        examples_text = "Here are some examples of simplified biomedical sentences:\n\n"
        for i, (complex_ex, simple_ex) in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Complex: {complex_ex}\n"
            examples_text += f"Simple: {simple_ex}\n\n"

    definitions_block = _definitions_block(definitions)
    definitions_rule = (
        "- Use the definitions below ONLY to simplify terms that appear in the sentence\n"
        if definitions
        else ""
    )

    user_prompt = (
        f"{examples_text}"
        "Now simplify this sentence following the same style:\n"
        "Rules:\n"
        "- Replace medical jargon with plain words\n"
        "- Remove statistical details (CI, p-values, RR, OR)\n"
        "- Keep all key facts and numbers\n"
        f"{definitions_rule}"
        "- Output ONLY the simplified sentence\n\n"
        f"{definitions_block}"
        f"Sentence: {input_sentence}\n"
        "Simplified:"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(DEFAULT_SYSTEM_PROMPT, user_prompt, tokenizer)

    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_rag_postedit_prompt(
    original_sentence: str,
    draft_simplification: str,
    definitions: List[Tuple[str, str]],
    tokenizer=None,
) -> str:
    """
    Create a retrieval-augmented post-editing prompt.

    Used as a second pass *after* the model has produced a draft simplification.
    Retrieved definitions correspond to technical terms that still appear in the
    draft, so the model can replace remaining jargon while staying faithful to
    the original sentence.

    Args:
        original_sentence: The original complex sentence (for faithfulness)
        draft_simplification: The model's first-pass simplification
        definitions: List of (term, definition) tuples retrieved from the draft
        tokenizer: Optional tokenizer for chat template formatting

    Returns:
        Formatted prompt string
    """
    definitions_block = ""
    if definitions:
        definitions_block = "Plain-language definitions for terms still in the draft:\n"
        for term, definition in definitions:
            definitions_block += f"- {term}: {definition}\n"
        definitions_block += "\n"

    user_prompt = (
        "Revise the draft simplification so a lay reader can understand it.\n"
        "Rules:\n"
        "- Replace any remaining medical jargon with plain words, using the definitions below\n"
        "- Do NOT add facts that are not in the original sentence\n"
        "- Keep all key facts and numbers from the original\n"
        "- If the draft is already clear, return it unchanged\n"
        "- Output ONLY the revised sentence\n\n"
        f"{definitions_block}"
        f"Original sentence: {original_sentence}\n"
        f"Draft simplification: {draft_simplification}\n"
        "Revised:"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(RAG_POSTEDIT_SYSTEM_PROMPT, user_prompt, tokenizer)

    return f"{RAG_POSTEDIT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_prompt(
    prompt_name: str,
    input_sentence: str,
    tokenizer=None,
    definitions: Optional[List[Tuple[str, str]]] = None,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Create a prompt based on the specified prompt variant.

    Args:
        prompt_name: Name of the prompt variant ('default_zero_shot' or 'few_shot')
        input_sentence: The complex sentence to simplify
        tokenizer: Optional tokenizer for chat template formatting
        definitions: Optional (term, definition) tuples to inject (RAG before mode)
        examples: Optional list of (complex, simple) tuples for few_shot

    Returns:
        Formatted prompt string
    """
    if prompt_name == "default_zero_shot":
        return create_default_prompt(input_sentence, tokenizer, definitions=definitions)
    elif prompt_name == "few_shot":
        return create_few_shot_prompt(
            input_sentence, examples or [], tokenizer, definitions=definitions
        )
    else:
        raise ValueError(
            f"Unknown prompt name: {prompt_name}. "
            f"Supported: 'default_zero_shot', 'few_shot'"
        )
