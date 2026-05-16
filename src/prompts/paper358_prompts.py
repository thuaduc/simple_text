"""
Prompt templates from LIS at SimpleText 2025 (Paper 358).

These prompts were used in their RAG-based simplification pipeline
that achieved a SARI score of 43.51 (5th place at CLEF 2025).

Reference: Paper 358 - "LIS at SimpleText 2025: Enhancing Scientific Text 
Accessibility with LLMs and Retrieval-Augmented Generation"
"""

from typing import List, Dict, Optional


def get_keyword_extraction_prompt(complex_text: str) -> str:
    """
    Keyword-Guided Retrieval Prompt (KGR) from Paper 358.
    
    Used to extract domain-specific keywords from complex text
    for definition retrieval.
    
    Args:
        complex_text: The complex scientific text to extract keywords from
        
    Returns:
        Formatted prompt for keyword extraction
    """
    prompt = f"""I have the following document: {complex_text}
Please give me the keywords that are present in this document and separate them with commas. Make sure you to only return the keywords and say nothing else. For example, don't say: "Here are the keywords present in the document"
"""
    return prompt


def get_zero_shot_prompt(complex_text: str, definitions: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Definition-Augmented Simplification Prompt (Zero-Shot) - DASP_0 from Paper 358.
    
    Best performing prompt that achieved SARI 43.51.
    Includes retrieved definitions but no examples.
    
    Args:
        complex_text: The complex text to simplify
        definitions: List of dicts with 'term' and 'definition' keys
        
    Returns:
        Formatted prompt for zero-shot simplification
    """
    definitions_text = ""
    if definitions:
        definitions_text = "\n".join([f"{d['term']}: {d['definition']}" for d in definitions])
        definitions_text = f"DEFINITIONS:\n{definitions_text}\n\n"
    
    prompt = f"""Using these definitions, please simplify the following scientific text for a general audience. Use plain language and explain any complex terms or acronyms. Ensure that all numbers, results, and facts remain exactly the same. Do not paraphrase numerical data or alter the meaning of findings.

{definitions_text}TEXT: {complex_text}
"""
    return prompt


def get_one_shot_prompt(
    complex_text: str,
    example_complex: str,
    example_simple: str,
    definitions: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Definition-Augmented Simplification Prompt (One-Shot) - DASP_1 from Paper 358.
    
    Includes one example alongside definitions.
    Achieved SARI 42.47 (slightly lower than zero-shot).
    
    Args:
        complex_text: The complex text to simplify
        example_complex: Example of complex text
        example_simple: Example of simplified text
        definitions: List of dicts with 'term' and 'definition' keys
        
    Returns:
        Formatted prompt for one-shot simplification
    """
    definitions_text = ""
    if definitions:
        definitions_text = "\n".join([f"{d['term']}: {d['definition']}" for d in definitions])
    
    prompt = f"""You are a helpful assistant that simplifies biomedical or scientific texts.

Task:
Using these definitions, simplify the following scientific text for a general audience. Use plain language and explain any complex terms or acronyms. Ensure that all numbers, results, and facts remain exactly the same. Do not paraphrase numerical data or alter the meaning of findings.

Example:
Definitions: {definitions_text if definitions else "N/A"}
Text: {example_complex}
Simplified: {example_simple}

Now do the same for the following:
Definitions: {definitions_text if definitions else "N/A"}
Text: {complex_text}
Simplified:"""
    return prompt


def get_iterative_refinement_prompt(complex_text: str, first_simplified: str) -> str:
    """
    Iterative Refinement Prompt (IRP) from Paper 358.
    
    Used in post-competition experiments to refine initial simplifications.
    Achieved SARI 43.10 (slight improvement over baseline).
    
    Args:
        complex_text: The original complex text
        first_simplified: The initial simplified version to refine
        
    Returns:
        Formatted prompt for iterative refinement
    """
    prompt = f"""Improve the simplified version of the scientific text below to make it clearer and easier for a general audience.

Your goal is to maximize the SARI score by simplifying language and structure, while keeping all facts, numbers, and findings exactly the same. Do this step by step.

ORIGINAL TEXT: {complex_text}

FIRST SIMPLIFIED VERSION: {first_simplified}

REFINED VERSION:"""
    return prompt


# Pre-configured prompt templates
PROMPT_TEMPLATES = {
    "KGR": get_keyword_extraction_prompt,
    "DASP_0": get_zero_shot_prompt,
    "DASP_1": get_one_shot_prompt,
    "IRP": get_iterative_refinement_prompt
}


def get_prompt(
    prompt_type: str,
    complex_text: str,
    definitions: Optional[List[Dict[str, str]]] = None,
    example_complex: Optional[str] = None,
    example_simple: Optional[str] = None,
    first_simplified: Optional[str] = None
) -> str:
    """
    Get a formatted prompt based on the Paper 358 prompt type.
    
    Args:
        prompt_type: One of "KGR", "DASP_0", "DASP_1", "IRP"
        complex_text: The complex text to process
        definitions: List of term-definition dicts (for DASP_0, DASP_1)
        example_complex: Example complex text (for DASP_1)
        example_simple: Example simplified text (for DASP_1)
        first_simplified: Initial simplification (for IRP)
        
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If prompt_type is not recognized or required args are missing
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Must be one of {list(PROMPT_TEMPLATES.keys())}")
    
    if prompt_type == "KGR":
        return get_keyword_extraction_prompt(complex_text)
    
    elif prompt_type == "DASP_0":
        return get_zero_shot_prompt(complex_text, definitions)
    
    elif prompt_type == "DASP_1":
        if example_complex is None or example_simple is None:
            raise ValueError("DASP_1 requires example_complex and example_simple")
        return get_one_shot_prompt(complex_text, example_complex, example_simple, definitions)
    
    elif prompt_type == "IRP":
        if first_simplified is None:
            raise ValueError("IRP requires first_simplified")
        return get_iterative_refinement_prompt(complex_text, first_simplified)


if __name__ == "__main__":
    # Test the prompts
    print("Testing Paper 358 Prompts\n")
    print("="*80)
    
    # Example data
    complex_text = "Resuscitation with a nasal interface may reduce the rate of intubation in the DR, but the evidence is very uncertain."
    definitions = [
        {"term": "intubation", "definition": "insertion of a breathing tube"},
        {"term": "DR", "definition": "delivery room"}
    ]
    
    # Test KGR
    print("\n1. Keyword Extraction (KGR):")
    print("-"*80)
    print(get_prompt("KGR", complex_text))
    
    # Test DASP_0
    print("\n2. Zero-Shot with Definitions (DASP_0):")
    print("-"*80)
    print(get_prompt("DASP_0", complex_text, definitions=definitions))
    
    # Test DASP_1
    print("\n3. One-Shot with Definitions (DASP_1):")
    print("-"*80)
    example_complex = "We included five trials, in which 1406 infants participated."
    example_simple = "We found five studies that involved 1406 babies."
    print(get_prompt("DASP_1", complex_text, definitions=definitions, 
                    example_complex=example_complex, example_simple=example_simple))
    
    # Test IRP
    print("\n4. Iterative Refinement (IRP):")
    print("-"*80)
    first_simplified = "Using a nose mask may reduce the need for breathing tubes, but we are not certain."
    print(get_prompt("IRP", complex_text, first_simplified=first_simplified))
