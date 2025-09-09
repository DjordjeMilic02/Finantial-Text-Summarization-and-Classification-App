import random
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix


CUSTOM_CB_PATH = "customDataset/parsed_custom_cb.jsonl"
MODEL_ID = "yiyanghkust/finbert-pretrain"
OUTPUT_DIR = "./finbert-finetuned"
BATCH_SIZE = 8
NUM_EPOCHS = 3
MAX_LENGTH = 512
NUM_LABELS = 3

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted')
    }

def preprocess_function(examples, tokenizer):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

def main():
    print("[INFO] Loading datasets...")

    news_ds = load_dataset("ashraq/financial-news-articles", split="train")
    news_ds = news_ds.shuffle(seed=42).select(range(3585))
    news_ds = news_ds.map(lambda x: {"text": x["text"], "label": 0})

    earnings1 = load_dataset("yeong-hwan/2024-earnings-call-transcript", split="train")
    def flatten_conversations(example):
        if isinstance(example["conversations"], list):
            return {"text": " ".join([turn["content"] for turn in example["conversations"]]), "label": 1}
        return {"text": "", "label": 1}
    earnings1 = earnings1.map(flatten_conversations)

    earnings2 = load_dataset("soumakchak/earnings_call_dataset", split="train")
    earnings2 = earnings2.map(lambda x: {"text": x["document"], "label": 1})
    earnings_ds = concatenate_datasets([earnings1, earnings2]).shuffle(seed=42).select(range(3585))

    central_ds = load_dataset("samchain/bis_central_bank_speeches", split="train")
    central_ds = central_ds.shuffle(seed=42).select(range(2249))
    central_ds = central_ds.map(lambda x: {"text": x["text"], "label": 2})

    custom_cb = load_dataset("json", data_files=CUSTOM_CB_PATH, split="train")
    custom_cb = custom_cb.map(lambda x: {"text": x["text"], "label": 2})

    dataset = concatenate_datasets([news_ds, earnings_ds, central_ds, custom_cb])
    dataset = dataset.shuffle(seed=42)

    print(f"[INFO] Total examples: {len(dataset)}")

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, test_ds = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        trust_remote_code=False,
        use_safetensors=True
    )

    def tokenize_fn(examples):
        return preprocess_function(examples, tokenizer)

    print("[INFO] Tokenizing dataset...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=300,
        logging_steps=100,
        save_strategy="steps",
        save_steps=300,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=12,
        weight_decay=0.02,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        optim="adamw_torch",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Evaluating on test set...")
    metrics = trainer.evaluate(eval_dataset=tokenized_test)
    print(metrics)

    preds_output = trainer.predict(tokenized_test)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))

    print("[INFO] Saving model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[SUCCESS] Fine-tuning complete! Model saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()