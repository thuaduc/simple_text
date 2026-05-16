"""Prompts and templates for text simplification."""

from .few_shot_examples import get_curated_examples
from .paper358_prompts import (
    get_keyword_extraction_prompt,
    get_zero_shot_prompt,
    get_one_shot_prompt,
    get_iterative_refinement_prompt,
    get_prompt,
    PROMPT_TEMPLATES
)

__all__ = [
    'get_curated_examples',
    'get_keyword_extraction_prompt',
    'get_zero_shot_prompt',
    'get_one_shot_prompt',
    'get_iterative_refinement_prompt',
    'get_prompt',
    'PROMPT_TEMPLATES'
]
