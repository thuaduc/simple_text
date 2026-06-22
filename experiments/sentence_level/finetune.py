"""LoRA fine-tuning for Qwen3.5 on rephrase-only sentence simplification.

Trains a LoRA adapter so the model maps a complex biomedical sentence
(using the default zero-shot prompt) to its simplified reference.
Loss is computed only on the simplified target, not the prompt.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from src.config import MODEL_NAME, DATA_DIR
from src.utils.data_loader import load_cochrane_sentences
from src.prompts.templates import create_default_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5 for sentence simplification")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=str(_SCRIPT_DIR / "lora_adapter"))
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--epochs', type=float, default=3.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--load_in_4bit', action='store_true', help='QLoRA: load base model in 4-bit')
    parser.add_argument('--early_stopping_patience', type=int, default=1, help='Stop after N epochs without val loss improvement')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_examples(split, data_dir, tokenizer, max_seq_len):
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
        prompt = create_default_prompt(complex_sent, tokenizer)
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

    print(f"Loading tokenizer/model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": torch.float16}
    if args.load_in_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_examples = build_examples("train", args.data_dir, tokenizer, args.max_seq_len)
    val_examples = build_examples("val", args.data_dir, tokenizer, args.max_seq_len)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
