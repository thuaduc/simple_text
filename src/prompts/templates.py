"""Prompt templates for sentence-level biomedical text simplification."""

from typing import List, Tuple, Optional


DEFAULT_SYSTEM_PROMPT = (
    "You simplify biomedical sentences. Be conservative. "
    "Keep the sentence structure when possible. Only change what "
    "is necessary to make it understandable to a general audience."
)

NIH_K8_SYSTEM_PROMPT = (
    "You are an expert at simplifying biomedical text for a general audience. "
    "Write at an 8th-grade reading level using plain language principles from "
    "the NIH Clear Communication guidelines. Make medical content accessible "
    "while staying faithful to the original meaning."
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


def create_default_prompt(input_sentence: str, tokenizer=None) -> str:
    """Create the default zero-shot simplification prompt."""
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
        return _chat_prompt_no_think(DEFAULT_SYSTEM_PROMPT, user_prompt, tokenizer)

    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_definition_augmented_prompt(
    input_sentence: str,
    definitions: List[Tuple[str, str]],
    tokenizer=None
) -> str:
    """
    Create a definition-augmented simplification prompt.

    Args:
        input_sentence: The complex sentence to simplify
        definitions: List of (term, definition) tuples from glossary retrieval
        tokenizer: Optional tokenizer for chat template formatting

    Returns:
        Formatted prompt string
    """
    # Build definitions block if any definitions are provided
    definitions_block = ""
    if definitions:
        definitions_block = "Relevant definitions:\n"
        for term, definition in definitions:
            definitions_block += f"- {term}: {definition}\n"
        definitions_block += "\n"

    user_prompt = (
        "Simplify this biomedical sentence for a lay reader.\n"
        "Rules:\n"
        "- Replace medical jargon with plain words\n"
        "- Remove statistical details (CI, p-values, RR, OR)\n"
        "- Keep all key facts and numbers\n"
        "- If already simple, return it unchanged\n"
        "- Do NOT add new information beyond what is in the sentence\n"
        "- Use the definitions below ONLY to simplify terms that appear in the sentence\n"
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


def create_nih_k8_prompt(input_sentence: str, tokenizer=None) -> str:
    """
    Create an NIH plain-language grade-8 guidance prompt.
    
    Emphasizes plain language principles and 8th-grade reading level.
    """
    user_prompt = (
        "Simplify this biomedical sentence for an 8th-grade reading level.\n"
        "Plain language guidelines:\n"
        "- Use common, everyday words (avoid jargon)\n"
        "- Use short sentences (15-20 words)\n"
        "- Use active voice\n"
        "- Remove statistics and technical details (CI, p-values, percentages)\n"
        "- Explain acronyms in plain English\n"
        "- Keep all key medical findings\n"
        "- Output ONLY the simplified sentence\n\n"
        f"Sentence: {input_sentence}\n"
        "Simplified:"
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(NIH_K8_SYSTEM_PROMPT, user_prompt, tokenizer)
    
    return f"{NIH_K8_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_plan_guided_prompt(input_sentence: str, tokenizer=None) -> str:
    """
    Create a two-stage plan-guided prompt.
    
    First asks for a simplification plan, then the simplified sentence.
    """
    user_prompt = (
        "Simplify this biomedical sentence in two steps:\n\n"
        "Step 1 - Plan: Identify what needs to be simplified:\n"
        "- Which medical terms need plain language?\n"
        "- Which details should be removed (statistics, jargon)?\n"
        "- What key information must be kept?\n\n"
        "Step 2 - Simplify: Write ONLY the simplified sentence based on your plan.\n\n"
        f"Sentence: {input_sentence}\n\n"
        "Plan:"
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(DEFAULT_SYSTEM_PROMPT, user_prompt, tokenizer)
    
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_few_shot_prompt(
    input_sentence: str,
    examples: List[Tuple[str, str]],
    tokenizer=None
) -> str:
    """
    Create a few-shot prompt with retrieved examples.
    
    Args:
        input_sentence: The sentence to simplify
        examples: List of (complex, simple) example pairs
        tokenizer: Optional tokenizer for chat template
    """
    examples_text = ""
    if examples:
        examples_text = "Here are some examples of simplified biomedical sentences:\n\n"
        for i, (complex_ex, simple_ex) in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Complex: {complex_ex}\n"
            examples_text += f"Simple: {simple_ex}\n\n"
    
    user_prompt = (
        f"{examples_text}"
        "Now simplify this sentence following the same style:\n"
        "Rules:\n"
        "- Replace medical jargon with plain words\n"
        "- Remove statistical details (CI, p-values, RR, OR)\n"
        "- Keep all key facts and numbers\n"
        "- Output ONLY the simplified sentence\n\n"
        f"Sentence: {input_sentence}\n"
        "Simplified:"
    )
    
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return _chat_prompt_no_think(DEFAULT_SYSTEM_PROMPT, user_prompt, tokenizer)
    
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"


def create_prompt(
    prompt_name: str,
    input_sentence: str,
    tokenizer=None,
    definitions: Optional[List[Tuple[str, str]]] = None,
    examples: Optional[List[Tuple[str, str]]] = None
) -> str:
    """
    Create a prompt based on the specified prompt variant.

    Args:
        prompt_name: Name of the prompt variant
        input_sentence: The complex sentence to simplify
        tokenizer: Optional tokenizer for chat template formatting
        definitions: Optional list of (term, definition) tuples for definition_augmented
        examples: Optional list of (complex, simple) tuples for few_shot

    Returns:
        Formatted prompt string
    """
    if prompt_name == "default_zero_shot":
        return create_default_prompt(input_sentence, tokenizer)
    elif prompt_name == "nih_k8":
        return create_nih_k8_prompt(input_sentence, tokenizer)
    elif prompt_name == "plan_guided":
        return create_plan_guided_prompt(input_sentence, tokenizer)
    elif prompt_name == "few_shot":
        if examples is None:
            examples = []
        return create_few_shot_prompt(input_sentence, examples, tokenizer)
    elif prompt_name == "definition_augmented":
        if definitions is None:
            definitions = []
        return create_definition_augmented_prompt(input_sentence, definitions, tokenizer)
    else:
        raise ValueError(
            f"Unknown prompt name: {prompt_name}. "
            f"Supported: 'default_zero_shot', 'nih_k8', 'plan_guided', 'few_shot', 'definition_augmented'"
        )

