"""LoRA fine-tuning for Qwen3.5 on rephrase-only sentence simplification.

Trains a LoRA adapter so the model maps a complex biomedical sentence
using the selected prompt style to its simplified reference.
Loss is computed only on the simplified target, not the prompt.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.config import DATA_DIR
from src.utils.data_loader import load_cochrane_sentences
from src.prompts.templates import create_prompt
from src.retrieval import FewShotRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5 for sentence simplification")
    parser.add_argument('--model', type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help="Early stopping patience (in eval calls). If unset, no early stopping.",
    )
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument(
        '--prompt',
        type=str,
        choices=['default_zero_shot', 'few_shot'],
        default='default_zero_shot',
        help='Prompt variant to fine-tune with'
    )
    parser.add_argument(
        '--num_shots',
        type=int,
        default=3,
        help='Number of retrieved examples for few_shot prompts'
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_examples(split, data_dir, tokenizer, max_seq_len, prompt_name, num_shots, retriever):
    """Tokenize each pair as prompt + target, masking prompt tokens in the labels."""
    complex_sents, references, _, _ = load_cochrane_sentences(
        split=split, data_dir=data_dir, rephrase_only=True
    )

    examples = []
    skipped_too_long = 0
    truncated_prompts = 0
    for complex_sent, refs in zip(complex_sents, references):
        if not refs:
            continue
        examples_for_prompt = None
        if prompt_name == "few_shot" and retriever is not None:
            examples_for_prompt = retriever.retrieve(
                complex_sent,
                k=num_shots,
                exclude_self=(split == "train")
            )
        prompt = create_prompt(
            prompt_name,
            complex_sent,
            tokenizer,
            examples=examples_for_prompt
        )
        target = refs[0].strip() + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        prompt_budget = max_seq_len - len(target_ids)
        if prompt_budget <= 0:
            skipped_too_long += 1
            continue

        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]
            truncated_prompts += 1

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        examples.append({"input_ids": input_ids, "labels": labels})

    print(f"Built {len(examples)} {split} examples")
    if truncated_prompts or skipped_too_long:
        print(
            f"  Truncated {truncated_prompts} prompts; "
            f"skipped {skipped_too_long} examples with targets longer than max_seq_len"
        )
    return examples


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print(
        f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules=all-linear"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    retriever = None
    if args.prompt == "few_shot":
        train_complex, train_references, _, _ = load_cochrane_sentences(
            split="train", data_dir=args.data_dir, rephrase_only=True
        )
        retriever = FewShotRetriever(train_complex, train_references, method="lexical")
        print(f"Using few_shot prompts with {args.num_shots} retrieved examples")

    train_examples = build_examples(
        "train",
        args.data_dir,
        tokenizer,
        args.max_length,
        args.prompt,
        args.num_shots,
        retriever
    )
    val_examples = build_examples(
        "val",
        args.data_dir,
        tokenizer,
        args.max_length,
        args.prompt,
        args.num_shots,
        retriever
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    use_early_stopping = args.patience is not None

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()

    print(f"Saving LoRA adapter to {args.output}")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))


if __name__ == "__main__":
    main()
