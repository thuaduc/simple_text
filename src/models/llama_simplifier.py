"""Model wrapper for text simplification."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

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
            # Default: conservative prompt based on CLEF 2025 winner analysis
            # Key findings: concise > verbose (THM), zero-shot > one-shot (LIS),
            # conservative > aggressive (SARI penalizes unwanted changes)
            system = (
                "You simplify biomedical sentences. Be conservative. "
                "Keep the sentence structure when possible. Only change what "
                "is necessary to make it understandable to a general audience."
            )
            
            user = (
                "Simplify this biomedical sentence for a lay reader.\n"
                "Rules:\n"
                "- Replace medical jargon with plain words\n"
                "- Remove statistical details (CI, p-values, RR, OR)\n"
                "- Keep all key facts and numbers\n"
                "- If already simple, return it unchanged\n"
                "- Do NOT add new information\n"
                "- Output ONLY the simplified sentence\n\n"
                f"Sentence: {complex_sentence}\n"
                "Simplified:"
            )
            
            # Use chat template if available, otherwise plain prompt
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"{system}\n\n{user}"
        
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
        for sentence in tqdm(
            complex_sentences,
            desc="Simplifying",
            unit="sent",
            mininterval=1.0,
        ):
            simplified.append(self.simplify(sentence, few_shot_examples))

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
