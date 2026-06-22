"""
src/reward_model/train.py

Fine-tunes distilBERT as a binary classifier -- our reward model.
"""

import sys
import random
from pathlib import Path

sys.path.insert(0, ".")

import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from src.reward_model.dataset import load_preference_pairs, build_training_records

SAVED_MODEL_PATH = "src/reward_model/saved_model"
BASE_MODEL       = "distilbert-base-uncased"
random.seed(42)


class RewardDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 256):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record  = self.records[idx]
        encoded = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels":         torch.tensor(record["label"], dtype=torch.long),
        }


def split_records(records, train_ratio=0.85):
    random.shuffle(records)
    split      = int(len(records) * train_ratio)
    train_recs = records[:split]
    val_recs   = records[split:]
    print(f"Training records  : {len(train_recs)}")
    print(f"Validation records: {len(val_recs)}")
    return train_recs, val_recs


def train_reward_model(records: list[dict]) -> None:
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL)
    model     = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
    )

    train_recs, val_recs = split_records(records)
    train_dataset = RewardDataset(train_recs, tokenizer)
    val_dataset   = RewardDataset(val_recs,   tokenizer)

    Path(SAVED_MODEL_PATH).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=SAVED_MODEL_PATH,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    print(f"Model saved to: {SAVED_MODEL_PATH}")


if __name__ == "__main__":
    pairs   = load_preference_pairs()
    records = build_training_records(pairs)
    train_reward_model(records)