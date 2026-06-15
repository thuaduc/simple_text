"""Model wrapper for sentence-level text simplification."""

import logging
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MAX_NEW_TOKENS, MODEL_NAME, TEMPERATURE
from src.prompts.templates import create_default_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceSimplifier:
    """Wrapper for causal language models used for sentence simplification."""

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
        Initialize the sentence simplification model.

        Args:
            model_name: HuggingFace model identifier (defaults to MODEL_NAME from .env)
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            load_in_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate (defaults to MAX_NEW_TOKENS from .env)
            temperature: Sampling temperature (defaults to TEMPERATURE from .env)
            do_sample: Whether to use sampling
        """
        self.model_name = model_name or MODEL_NAME
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        self.temperature = temperature or TEMPERATURE
        self.do_sample = do_sample

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
    ) -> str:
        """
        Simplify a single sentence.

        Args:
            complex_sentence: The complex sentence to simplify

        Returns:
            Simplified sentence
        """
        prompt = create_default_prompt(complex_sentence, self.tokenizer)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        simplified = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if "\n\n" in simplified:
            simplified = simplified.split("\n\n")[0].strip()

        return simplified

    def simplify_batch(
        self,
        complex_sentences: List[str],
        batch_size: int = 8
    ) -> List[str]:
        """
        Simplify multiple sentences in batches.

        Args:
            complex_sentences: List of complex sentences
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
            simplified.append(self.simplify(sentence))

        return simplified


if __name__ == "__main__":
    print("Testing Sentence Simplifier...")

    simplifier = SentenceSimplifier(
        load_in_4bit=torch.cuda.is_available()
    )

    test_sentence = "Resuscitation with a nasal interface may reduce the rate of intubation in the DR, but the evidence is very uncertain."

    print(f"\nInput: {test_sentence}")
    print("\nSimplifying...")

    simplified = simplifier.simplify(test_sentence)

    print(f"\nOutput: {simplified}")
