"""Model wrapper for text simplification."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

from src.config import MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaSimplifier:
    """Wrapper for language models for text simplification."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: bool = True
    ):
        """
        Initialize the text simplification model.
        
        Default model: Mistral 7B Instruct (best performing in Paper 358: SARI 43.51)
        Configuration is loaded from .env file by default.
        
        Args:
            model_name: HuggingFace model identifier (defaults to MODEL_NAME from .env)
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            load_in_4bit: Whether to use 4-bit quantization (recommended for 7B models)
            max_new_tokens: Maximum tokens to generate (defaults to MAX_NEW_TOKENS from .env)
            temperature: Sampling temperature (defaults to TEMPERATURE from .env)
            do_sample: Whether to use sampling
        """
        # Use config values if not provided
        self.model_name = model_name or MODEL_NAME
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        self.temperature = temperature or TEMPERATURE
        self.do_sample = do_sample
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading {self.model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            
            logger.info("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
        
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    
    def simplify(
        self,
        complex_sentence: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        prompt_type: str = "default",
        definitions: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Simplify a single sentence.
        
        Args:
            complex_sentence: The complex sentence to simplify
            few_shot_examples: Optional list of example dictionaries with 'complex' and 'simple' keys
            prompt_type: Type of prompt ("default", "paper358_zero_shot", "paper358_one_shot")
            definitions: Optional list of term-definition dicts for Paper 358 prompts
        
        Returns:
            Simplified sentence
        """
        # Build prompt based on type
        if prompt_type == "paper358_zero_shot":
            from src.prompts.paper358_prompts import get_zero_shot_prompt
            prompt = get_zero_shot_prompt(complex_sentence, definitions)
        elif prompt_type == "paper358_one_shot" and few_shot_examples:
            from src.prompts.paper358_prompts import get_one_shot_prompt
            example = few_shot_examples[0]
            prompt = get_one_shot_prompt(
                complex_sentence,
                example['complex'],
                example['simple'],
                definitions
            )
        else:
            # Default: instruction phrasing adapted from Paper 358 (DASP_0/DASP_1, SARI 43.51)
            prompt = (
                "You are a helpful assistant that simplifies biomedical or scientific texts.\n\n"
                "Task: Simplify the following scientific text for a general audience. "
                "Use plain language and explain any complex terms or acronyms. "
                "Ensure that all numbers, results, and facts remain exactly the same. "
                "Do not paraphrase numerical data or alter the meaning of findings.\n\n"
            )
            
            if few_shot_examples:
                for ex in few_shot_examples:
                    prompt += f"Example:\nText: {ex['complex']}\nSimplified: {ex['simple']}\n\n"
                prompt += "Now do the same for the following:\n"
            
            prompt += f"Text: {complex_sentence}\nSimplified:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        simplified = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Clean up: take only first sentence/paragraph if multiple generated
        if "\n\n" in simplified:
            simplified = simplified.split("\n\n")[0].strip()
        
        return simplified
    
    def simplify_batch(
        self,
        complex_sentences: List[str],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        batch_size: int = 8
    ) -> List[str]:
        """
        Simplify multiple sentences in batches.
        
        Args:
            complex_sentences: List of complex sentences
            few_shot_examples: Optional list of example dictionaries
            batch_size: Batch size for processing
        
        Returns:
            List of simplified sentences
        """
        simplified = []
        
        for i in range(0, len(complex_sentences), batch_size):
            batch = complex_sentences[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(complex_sentences)-1)//batch_size + 1}")
            
            for sentence in batch:
                simplified_sent = self.simplify(sentence, few_shot_examples)
                simplified.append(simplified_sent)
        
        return simplified


if __name__ == "__main__":
    # Test the simplifier
    print("Testing Llama Simplifier...")
    
    # Create simplifier instance
    simplifier = LlamaSimplifier(
        load_in_4bit=torch.cuda.is_available()
    )
    
    # Test examples
    examples = [
        {
            'complex': 'We included five trials, in which 1406 infants participated.',
            'simple': 'We found five studies that involved 1406 babies.'
        }
    ]
    
    test_sentence = "Resuscitation with a nasal interface may reduce the rate of intubation in the DR, but the evidence is very uncertain."
    
    print(f"\nInput: {test_sentence}")
    print("\nSimplifying...")
    
    simplified = simplifier.simplify(test_sentence, few_shot_examples=examples)
    
    print(f"\nOutput: {simplified}")
