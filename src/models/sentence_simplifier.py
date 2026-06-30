"""Model wrapper for sentence-level text simplification."""

import logging
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MAX_NEW_TOKENS, MODEL_NAME, TEMPERATURE
from src.prompts.templates import create_prompt, create_rag_postedit_prompt

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
        do_sample: bool = True,
        prompt_name: str = "default_zero_shot",
        adapter_path: Optional[str] = None,
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
            prompt_name: Prompt variant to use ('default_zero_shot' or 'few_shot')
        """
        self.model_name = model_name or MODEL_NAME
        self.max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        self.temperature = temperature or TEMPERATURE
        self.do_sample = do_sample
        self.prompt_name = prompt_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading {self.model_name} on {self.device}...")

        tokenizer_source = adapter_path or self.model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except OSError:
            if adapter_path:
                logger.info(
                    f"Tokenizer not found at {adapter_path}; falling back to base tokenizer {self.model_name}."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                raise
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            logger.info("Loading with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.model.to(self.device)

        if adapter_path:
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        logger.info("Model loaded successfully!")

    def simplify(
        self,
        complex_sentence: str,
        definitions: Optional[List] = None,
        examples: Optional[List] = None,
    ) -> str:
        """
        Simplify a single sentence.

        Args:
            complex_sentence: The complex sentence to simplify
            definitions: Optional list of (term, definition) tuples to inject (RAG before mode)
            examples: Optional list of (complex, simple) tuples for few_shot

        Returns:
            Simplified sentence
        """
        prompt = create_prompt(
            self.prompt_name,
            complex_sentence,
            self.tokenizer,
            definitions=definitions,
            examples=examples,
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        simplified = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if "\n\n" in simplified:
            simplified = simplified.split("\n\n")[0].strip()

        return simplified

    def simplify_batch(
        self,
        complex_sentences: List[str],
        batch_size: int = 8,
        definitions_list: Optional[List[List]] = None,
        examples_list: Optional[List[List]] = None,
    ) -> List[str]:
        """
        Simplify multiple sentences in batches.

        Args:
            complex_sentences: List of complex sentences
            batch_size: Batch size for processing
            definitions_list: Optional list of definition lists (one per sentence)
            examples_list: Optional list of example lists (one per sentence)

        Returns:
            List of simplified sentences
        """
        if definitions_list is None:
            definitions_list = [None] * len(complex_sentences)
        if examples_list is None:
            examples_list = [None] * len(complex_sentences)

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if len(definitions_list) != len(complex_sentences):
            raise ValueError(
                f"definitions_list length ({len(definitions_list)}) must match "
                f"complex_sentences length ({len(complex_sentences)})"
            )

        if len(examples_list) != len(complex_sentences):
            raise ValueError(
                f"examples_list length ({len(examples_list)}) must match "
                f"complex_sentences length ({len(complex_sentences)})"
            )

        simplified = []
        for start in tqdm(
            range(0, len(complex_sentences), batch_size),
            desc="Simplifying",
            unit="batch",
            mininterval=1.0,
            total=(len(complex_sentences) + batch_size - 1) // batch_size,
        ):
            end = start + batch_size
            prompts = [
                create_prompt(
                    self.prompt_name,
                    sentence,
                    self.tokenizer,
                    definitions=definitions,
                    examples=examples,
                )
                for sentence, definitions, examples in zip(
                    complex_sentences[start:end],
                    definitions_list[start:end],
                    examples_list[start:end],
                )
            ]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            prompt_length = inputs.input_ids.shape[1]
            for output in outputs:
                new_tokens = output[prompt_length:]
                text = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                if "\n\n" in text:
                    text = text.split("\n\n")[0].strip()
                simplified.append(text)

        return simplified

    def simplify_candidates_batch(
        self,
        complex_sentences: List[str],
        num_candidates: int,
        batch_size: int = 8,
        temperature: Optional[float] = None,
        definitions_list: Optional[List[List]] = None,
        examples_list: Optional[List[List]] = None,
        return_scores: bool = False,
    ):
        """Generate ``num_candidates`` sampled simplifications per sentence.

        Used by the candidate-generation + reranking pipeline. Sampling is
        forced on here (regardless of ``self.do_sample``) so the candidate pool
        is diverse; pass ``temperature`` to control diversity.

        Args:
            return_scores: if True, also return per-candidate length-normalized
                sequence log-probabilities (the model-confidence feature used by
                the learned reranker).

        Returns:
            If ``return_scores`` is False: a list (one per sentence) of
            candidate-string lists of length ``num_candidates``.
            If True: a tuple ``(candidates, scores)`` with matching shapes,
            where ``scores[i][c]`` is the mean token log-prob of candidate c.
        """
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if definitions_list is None:
            definitions_list = [None] * len(complex_sentences)
        if examples_list is None:
            examples_list = [None] * len(complex_sentences)
        if len(definitions_list) != len(complex_sentences):
            raise ValueError("definitions_list length must match complex_sentences")
        if len(examples_list) != len(complex_sentences):
            raise ValueError("examples_list length must match complex_sentences")

        temp = temperature if temperature is not None else self.temperature
        candidates: List[List[str]] = []
        scores: List[List[float]] = []
        for start in tqdm(
            range(0, len(complex_sentences), batch_size),
            desc=f"Sampling x{num_candidates}",
            unit="batch",
            mininterval=1.0,
            total=(len(complex_sentences) + batch_size - 1) // batch_size,
        ):
            end = start + batch_size
            prompts = [
                create_prompt(
                    self.prompt_name,
                    sentence,
                    self.tokenizer,
                    definitions=definitions,
                    examples=examples,
                )
                for sentence, definitions, examples in zip(
                    complex_sentences[start:end],
                    definitions_list[start:end],
                    examples_list[start:end],
                )
            ]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": temp,
                "num_return_sequences": num_candidates,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if return_scores:
                generate_kwargs["output_scores"] = True
                generate_kwargs["return_dict_in_generate"] = True

            with torch.no_grad():
                gen = self.model.generate(**inputs, **generate_kwargs)

            sequences = gen.sequences if return_scores else gen
            prompt_length = inputs.input_ids.shape[1]
            n_prompts = len(prompts)

            seq_logprobs = None
            if return_scores:
                # Length-normalized mean token log-prob per returned sequence.
                transition = self.model.compute_transition_scores(
                    gen.sequences, gen.scores, normalize_logits=True
                )
                gen_tokens = sequences[:, prompt_length:]
                mask = gen_tokens != self.tokenizer.pad_token_id
                tok_counts = mask.sum(dim=1).clamp(min=1)
                summed = (transition * mask).sum(dim=1)
                seq_logprobs = (summed / tok_counts).tolist()

            # sequences is (n_prompts * num_candidates, seq_len), grouped per prompt.
            for p in range(n_prompts):
                cand_texts: List[str] = []
                cand_scores: List[float] = []
                for c in range(num_candidates):
                    flat = p * num_candidates + c
                    new_tokens = sequences[flat][prompt_length:]
                    text = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    if "\n\n" in text:
                        text = text.split("\n\n")[0].strip()
                    cand_texts.append(text)
                    if return_scores:
                        cand_scores.append(seq_logprobs[flat])
                candidates.append(cand_texts)
                if return_scores:
                    scores.append(cand_scores)

        if return_scores:
            return candidates, scores
        return candidates

    def postedit_batch(
        self,
        original_sentences: List[str],
        draft_simplifications: List[str],
        definitions_list: List[List],
        batch_size: int = 8,
    ) -> List[str]:
        """
        Run a retrieval-augmented post-editing pass over draft simplifications.

        For each draft, the provided definitions correspond to technical terms
        still present in that draft. Drafts whose definition list is empty are
        returned unchanged (no second-pass generation), which saves compute and
        avoids needless drift on already-simple sentences.

        Args:
            original_sentences: The original complex sentences (for faithfulness)
            draft_simplifications: First-pass model outputs to revise
            definitions_list: List of (term, definition) lists, one per draft
            batch_size: Batch size for processing

        Returns:
            List of revised simplifications (same length/order as the input)
        """
        n = len(original_sentences)
        if not (len(draft_simplifications) == len(definitions_list) == n):
            raise ValueError(
                "original_sentences, draft_simplifications, and definitions_list "
                f"must have equal length (got {n}, {len(draft_simplifications)}, "
                f"{len(definitions_list)})"
            )

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        revised: List[str] = list(draft_simplifications)

        # Only post-edit drafts that still contain retrievable jargon.
        edit_indices = [i for i in range(n) if definitions_list[i]]
        if not edit_indices:
            return revised

        for start in tqdm(
            range(0, len(edit_indices), batch_size),
            desc="Post-editing",
            unit="batch",
            mininterval=1.0,
            total=(len(edit_indices) + batch_size - 1) // batch_size,
        ):
            batch_idx = edit_indices[start : start + batch_size]
            prompts = [
                create_rag_postedit_prompt(
                    original_sentences[i],
                    draft_simplifications[i],
                    definitions_list[i],
                    self.tokenizer,
                )
                for i in batch_idx
            ]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            prompt_length = inputs.input_ids.shape[1]
            for i, output in zip(batch_idx, outputs):
                new_tokens = output[prompt_length:]
                text = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                if "\n\n" in text:
                    text = text.split("\n\n")[0].strip()
                # Guard against empty post-edit collapsing a valid draft.
                if text:
                    revised[i] = text

        return revised


if __name__ == "__main__":
    print("Testing Sentence Simplifier...")

    simplifier = SentenceSimplifier(load_in_4bit=torch.cuda.is_available())

    test_sentence = "Resuscitation with a nasal interface may reduce the rate of intubation in the DR, but the evidence is very uncertain."

    print(f"\nInput: {test_sentence}")
    print("\nSimplifying...")

    simplified = simplifier.simplify(test_sentence)

    print(f"\nOutput: {simplified}")
