import os
import argparse
from typing import Dict, Any, List
from collections import Counter
import numpy as np

from datasets import load_dataset, DatasetDict, concatenate_datasets
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from transformers.trainer import Trainer as HFTrainer


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--output_dir", type=str, default="./cb-stance-flare")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--weight_loss", action="store_true", help="Use class-weighted CE (imbalanced data).")
    p.add_argument("--focal", action="store_true", help="Use focal loss (with class weights if --weight_loss).")
    return p.parse_args()


LABEL_CANON = ["Hawkish", "Neutral", "Dovish"]

def canonicalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    z = x.strip().lower()
    if z in ["hawk","hawkish","h"]: return "Hawkish"
    if z in ["dove","dovish","d"]: return "Dovish"
    if z in ["neutral","n","neut"]: return "Neutral"
    return x.strip().capitalize()

def make_label_maps(labels: List[str]):
    uniq = sorted(set(labels), key=lambda s: LABEL_CANON.index(s) if s in LABEL_CANON else 99)
    label2id = {lbl: i for i, lbl in enumerate(uniq)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label

def is_valid_row(ex: Dict[str, Any]) -> bool:
    t, y = ex.get("text", None), ex.get("label", None)
    return isinstance(t, str) and isinstance(y, str) and t.strip() != "" and y.strip() != ""


def load_flare_fomc_all():
    """Load retarfi/flare-fomc and normalize to columns: text, label."""
    ds = load_dataset("retarfi/flare-fomc")
    outs = []
    def norm(ex):
        return {"text": ex.get("text",""), "label": canonicalize_label(ex.get("answer",""))}
    for split in ds.keys():
        d = ds[split].map(norm, remove_columns=[c for c in ds[split].column_names if c not in ["text","label"]])
        d = d.filter(is_valid_row)
        outs.append(d)
    return concatenate_datasets(outs)


def build_metrics():
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            **acc.compute(predictions=preds, references=labels),
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    return compute_metrics


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction="mean"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedTrainer(HFTrainer):
    def __init__(self, class_weights=None, focal=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
        self.loss_fct = FocalLoss(alpha=alpha) if focal else nn.CrossEntropyLoss(weight=alpha)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def make_weighted_sampler(dataset, num_labels):
    labels = [int(ex["labels"]) for ex in dataset]
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    inv = 1.0 / np.clip(counts, 1, None)
    weights = [inv[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def main():
    args = build_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TRANSFORMERS_USE_SAFE_TENSORS", "1")

    all_ds = load_flare_fomc_all()
    try:
        s1 = all_ds.train_test_split(test_size=args.test_ratio, seed=args.seed, stratify_by_column="label")
    except Exception:
        s1 = all_ds.train_test_split(test_size=args.test_ratio, seed=args.seed)
    train_pool, test_ds = s1["train"], s1["test"]

    try:
        s2 = train_pool.train_test_split(test_size=args.val_ratio, seed=args.seed, stratify_by_column="label")
    except Exception:
        s2 = train_pool.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds, val_ds = s2["train"], s2["test"]
    ds = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

    print("Sizes ->", {k: len(v) for k, v in ds.items()})
    print("Label counts (train):", Counter([ex["label"] for ex in ds["train"]]))
    print("Label counts (val):  ", Counter([ex["label"] for ex in ds["validation"]]))
    print("Label counts (test): ", Counter([ex["label"] for ex in ds["test"]]))

    train_labels = [ex["label"] for ex in ds["train"]]
    label2id, id2label = make_label_maps(train_labels)

    def to_ids(example):
        example["labels"] = label2id.get(example["label"], -1)
        return example
    ds = DatasetDict({k: v.map(to_ids) for k, v in ds.items()})
    ds = DatasetDict({k: v.filter(lambda ex: ex["labels"] >= 0) for k, v in ds.items()})

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length, padding=False)
    tokenized = DatasetDict({
        k: v.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in v.column_names if c not in ["labels"]],
        )
        for k, v in ds.items()
    })

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    class_weights = None
    if args.weight_loss:
        counts = np.bincount([int(ex["labels"]) for ex in tokenized["train"]], minlength=num_labels).astype(float)
        inv = 1.0 / np.clip(counts, 1, None)
        class_weights = (inv / inv.sum()) * num_labels

    train_sampler = make_weighted_sampler(tokenized["train"], num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    metrics_fn = build_metrics()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    TrainerClass = WeightedTrainer if (args.weight_loss or args.focal) else Trainer
    if TrainerClass is WeightedTrainer:
        trainer = TrainerClass(
            class_weights=class_weights, focal=args.focal,
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tok,
            data_collator=data_collator,
            compute_metrics=metrics_fn,
        )
    else:
        trainer = TrainerClass(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tok,
            data_collator=data_collator,
            compute_metrics=metrics_fn,
        )

    def _get_train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    trainer.get_train_dataloader = _get_train_dataloader.__get__(trainer, type(trainer))

    trainer.train()

    test_out = trainer.predict(tokenized["test"])
    print("\n=== Test metrics ===")
    for k, v in test_out.metrics.items():
        if k.startswith("test_"):
            print(f"{k}: {v:.4f}")

    preds = np.argmax(test_out.predictions, axis=-1)
    refs = test_out.label_ids

    cm = np.zeros((num_labels, num_labels), dtype=int)
    for p, r in zip(preds, refs):
        cm[r, p] += 1

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    header = ["{:>10}".format("")] + ["{:>10}".format(model.config.id2label[i]) for i in range(num_labels)]
    print(" ".join(header))
    for i in range(num_labels):
        row = ["{:>10}".format(model.config.id2label[i])] + ["{:>10}".format(cm[i, j]) for j in range(num_labels)]
        print(" ".join(row))

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    print("\n=== Row-normalized Confusion Matrix (recall per class) ===")
    print(" ".join(header))
    for i in range(num_labels):
        row = ["{:>10}".format(model.config.id2label[i])] + ["{:>10.2f}".format(cm_norm[i, j]) for j in range(num_labels)]
        print(" ".join(row))

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
