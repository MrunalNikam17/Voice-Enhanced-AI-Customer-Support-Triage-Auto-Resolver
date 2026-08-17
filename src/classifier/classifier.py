"""
src/classifier/classifier.py

Fine-tunes DistilBERT to classify customer support tickets into categories
(using the Bitext dataset's `category` column), and provides a
ClassifierAgent class for inference once trained.

Training: run this file directly (ideally on Colab GPU).
    python -m src.classifier.classifier

Inference (used by the LangGraph workflow):
    from src.classifier.classifier import ClassifierAgent
    agent = ClassifierAgent()
    result = agent.predict("I was charged twice this month")
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

sys.path.append(os.getcwd())
from src.utils.config import config


# -----------------------------
# Dataset wrapper
# -----------------------------

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# -----------------------------
# Data loading / prep
# -----------------------------

def load_bitext(csv_path: str = None) -> pd.DataFrame:
    csv_path = csv_path or config.BITEXT_CSV_PATH
    df = pd.read_csv(csv_path)

    # Bitext columns: flags, instruction, category, intent, response
    df = df[["instruction", "category"]].dropna()
    df = df.rename(columns={"instruction": "text", "category": "label"})

    # Keep only categories we actually use (in case CSV has extras)
    df = df[df["label"].isin(config.CATEGORIES)].reset_index(drop=True)
    return df


def build_label_maps(labels: list):
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


# -----------------------------
# Training
# -----------------------------

def train():
    print("Loading data...")
    df = load_bitext()
    print(f"Loaded {len(df)} labeled tickets across {df['label'].nunique()} categories")
    print(df["label"].value_counts())

    label2id, id2label = build_label_maps(df["label"].tolist())
    df["label_id"] = df["label"].map(label2id)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE_MODEL)

    train_ds = TicketDataset(train_df["text"], train_df["label_id"], tokenizer, config.CLASSIFIER_MAX_LEN)
    val_ds = TicketDataset(val_df["text"], val_df["label_id"], tokenizer, config.CLASSIFIER_MAX_LEN)
    test_ds = TicketDataset(test_df["text"], test_df["label_id"], tokenizer, config.CLASSIFIER_MAX_LEN)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    args = TrainingArguments(
        output_dir=config.CLASSIFIER_MODEL_DIR,
        num_train_epochs=config.CLASSIFIER_EPOCHS,
        per_device_train_batch_size=config.CLASSIFIER_BATCH_SIZE,
        per_device_eval_batch_size=config.CLASSIFIER_BATCH_SIZE,
        learning_rate=config.CLASSIFIER_LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to=[],  # set to ["wandb"] if you want experiment tracking
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on held-out test set...")
    test_results = trainer.predict(test_ds)
    preds = np.argmax(test_results.predictions, axis=1)
    true = test_df["label_id"].tolist()

    print(classification_report(true, preds, target_names=[id2label[i] for i in range(len(id2label))]))

    # Save final model + tokenizer + label maps
    os.makedirs(config.CLASSIFIER_MODEL_DIR, exist_ok=True)
    trainer.save_model(config.CLASSIFIER_MODEL_DIR)
    tokenizer.save_pretrained(config.CLASSIFIER_MODEL_DIR)
    with open(os.path.join(config.CLASSIFIER_MODEL_DIR, "label_maps.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\nModel saved to {config.CLASSIFIER_MODEL_DIR}")


# -----------------------------
# Inference wrapper (used by the LangGraph workflow)
# -----------------------------

class ClassifierAgent:
    def __init__(self, model_dir: str = None):
        model_dir = model_dir or config.CLASSIFIER_MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        with open(os.path.join(model_dir, "label_maps.json")) as f:
            maps = json.load(f)
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=config.CLASSIFIER_MAX_LEN, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        pred_id = int(torch.argmax(probs))
        return {
            "category": self.id2label[pred_id],
            "confidence": float(probs[pred_id]),
        }


if __name__ == "__main__":
    train()