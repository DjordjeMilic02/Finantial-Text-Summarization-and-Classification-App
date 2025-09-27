from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import numpy as np
import os
import argparse
import torch
from typing import Dict, Any

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="t5-base")
    p.add_argument("--dataset_id", type=str, default="SelmaNajih001/CentralBanksSpeeches-Summary")
    p.add_argument("--output_dir", type=str, default="./t5-cb-speeches")
    p.add_argument("--max_input_length", type=int, default=1024)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--quick_test", action="store_true", help="Tiny slices")
    return p.parse_args()

def is_valid_example(ex: Dict[str, Any]) -> bool:
    t = ex.get("text", None)
    s = ex.get("Summary", None)
    return (
        isinstance(t, str) and isinstance(s, str)
        and t.strip() != "" and s.strip() != ""
    )


def preprocess_batch(examples: Dict[str, Any], tokenizer, max_input_length=1024, max_target_length=128):
    inputs_raw = [x if isinstance(x, str) else "" for x in examples.get("text", [])]
    targets_raw = [x if isinstance(x, str) else "" for x in examples.get("Summary", [])]

    inputs = [f"summarize: {t.strip()}" for t in inputs_raw]
    targets = [s.strip() for s in targets_raw]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )["input_ids"]

    labels = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs

def build_compute_metrics(tokenizer, rouge):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        scores = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {k: round(v * 100, 2) for k, v in scores.items()}
    return compute_metrics

def main():
    args = build_args()
    os.environ.setdefault("WANDB_DISABLED", "true")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ds_raw = load_dataset(args.dataset_id)

    ds_clean = {split: d.filter(is_valid_example) for split, d in ds_raw.items()}

    split_train = ds_clean["train"].train_test_split(test_size=0.1, seed=42)
    full = {
        "train": split_train["train"],
        "validation": split_train["test"],
        "test": ds_clean["test"],
    }

    if args.quick_test:
        full["train"] = full["train"].select(range(min(200, len(full["train"]))))
        full["validation"] = full["validation"].select(range(min(50, len(full["validation"]))))
        full["test"] = full["test"].select(range(min(50, len(full["test"]))))

    preprocess = lambda batch: preprocess_batch(
        batch, tokenizer, args.max_input_length, args.max_target_length
    )

    remove_cols = list(set(full["train"].column_names) - {"text", "Summary"})
    tokenized = {
        split: d.map(
            preprocess,
            batched=True,
            remove_columns=remove_cols,
        )
        for split, d in full.items()
    }

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    rouge = evaluate.load("rouge")
    compute_metrics = build_compute_metrics(tokenizer, rouge)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=500 if not args.quick_test else 100,
        save_steps=500 if not args.quick_test else 100,
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.beam_size,
        fp16=args.fp16,
        save_total_limit=2,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
        optim="adafactor",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        if k.startswith("test_"):
            print(f"{k}: {v}")

    if len(full["test"]) > 0:
        sample_idx = 0
        raw_doc = full["test"][sample_idx]["text"]
        inputs = tokenizer(
            [f"summarize: {raw_doc}"],
            max_length=args.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        summary_ids = model.generate(
            **inputs,
            max_length=args.max_target_length,
            num_beams=args.beam_size,
            length_penalty=0.9,
            early_stopping=True,
        )
        print("\n=== Sample summary ===")
        print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
