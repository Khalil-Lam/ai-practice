import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


class PoetryDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.items = []

        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        for row in rows:
            prompt = row["prompt"]
            answer = row["completion"] + tokenizer.eos_token

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
            answer_ids = tokenizer(
                answer,
                add_special_tokens=False,
            )["input_ids"]

            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            attention_mask = [1] * len(input_ids)

            # 如果截断后一个目标 token 都没剩下，这条样本没有训练价值。
            if all(label == -100 for label in labels):
                continue

            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PoetryCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"] + [self.pad_token_id] * pad_len
            )
            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )
            labels.append(
                item["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--dev", default="data/dev.jsonl")
    parser.add_argument("--output_dir", default="outputs/tang-poetry-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = PoetryDataset(
        args.train,
        tokenizer,
        args.max_length,
    )
    dev_dataset = PoetryDataset(
        args.dev,
        tokenizer,
        args.max_length,
    )

    print("Train samples:", len(train_dataset))
    print("Dev samples:", len(dev_dataset))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=(
            torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()
        ),
        bf16=(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=PoetryCollator(tokenizer),
    )

    trainer.train()

    Path(args.output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
