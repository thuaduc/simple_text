"""Prompt templates for text simplification."""

from typing import List, Dict


def create_few_shot_prompt(
    examples: List[Dict[str, str]],
    input_sentence: str
) -> str:
    """
    Create a few-shot prompt for text simplification.
    
    Args:
        examples: List of example dictionaries with 'complex' and 'simple' keys
        input_sentence: The complex sentence to simplify
    
    Returns:
        Formatted prompt string
    """
    prompt = "You are an expert at simplifying biomedical text for general audiences.\n\n"
    prompt += "Your task is to simplify complex medical and scientific sentences while:\n"
    prompt += "- Keeping the core meaning intact\n"
    prompt += "- Using simpler vocabulary\n"
    prompt += "- Breaking down technical terms\n"
    prompt += "- Making it accessible to non-experts\n\n"
    
    prompt += "Here are some examples:\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Complex: {example['complex']}\n"
        prompt += f"Simple: {example['simple']}\n\n"
    
    prompt += "Now simplify this sentence:\n"
    prompt += f"Complex: {input_sentence}\n"
    prompt += "Simple:"
    
    return prompt


def create_instruction_prompt(input_sentence: str) -> str:
    """
    Create a zero-shot instruction prompt.
    
    Args:
        input_sentence: The complex sentence to simplify
    
    Returns:
        Formatted prompt string
    """
    prompt = "Simplify the following medical text for a general audience. "
    prompt += "Use simpler vocabulary and explain technical terms.\n\n"
    prompt += f"Complex: {input_sentence}\n"
    prompt += "Simple:"
    
    return prompt


def format_chat_prompt(
    examples: List[Dict[str, str]],
    input_sentence: str
) -> List[Dict[str, str]]:
    """
    Format prompt for chat-based models (like Llama Instruct).
    
    Args:
        examples: List of example dictionaries
        input_sentence: The complex sentence to simplify
    
    Returns:
        List of message dictionaries for chat format
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at simplifying biomedical and medical text "
                "for general audiences. You make complex medical concepts accessible "
                "to non-experts while preserving accuracy."
            )
        }
    ]
    
    # Add examples as user-assistant pairs
    for example in examples:
        messages.append({
            "role": "user",
            "content": f"Simplify this: {example['complex']}"
        })
        messages.append({
            "role": "assistant",
            "content": example['simple']
        })
    
    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"Simplify this: {input_sentence}"
    })
    
    return messages


if __name__ == "__main__":
    # Test the templates
    examples = [
        {
            'complex': 'We included five trials, in which 1406 infants participated.',
            'simple': 'We found five studies that involved 1406 babies.'
        },
        {
            'complex': 'The evidence is very uncertain.',
            'simple': 'We are not sure about the results.'
        }
    ]
    
    test_sentence = "Resuscitation with a nasal interface may reduce the rate of intubation."
    
    print("Few-shot prompt:")
    print(create_few_shot_prompt(examples, test_sentence))
    print("\n" + "="*80 + "\n")
    
    print("Chat format:")
    messages = format_chat_prompt(examples, test_sentence)
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content']}")
        print()
