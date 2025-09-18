from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import numpy as np
import os
import argparse
import torch

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="google/pegasus-large")
    p.add_argument("--dataset_id", type=str, default="soumakchak/earnings_call_dataset")
    p.add_argument("--output_dir", type=str, default="./pegasus-earnings-fast")

    p.add_argument("--max_input_length", type=int, default=768)
    p.add_argument("--max_target_length", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--quick_test", action="store_true", help="Minisized run for smoke testing")

    p.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder params (faster)")
    p.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Tokenization workers")
    return p.parse_args()

def preprocess_batch(examples, tokenizer, max_input_length=768, max_target_length=96):
    inputs = examples["document"]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

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

def maybe_freeze_encoder(model, freeze: bool):
    if not freeze:
        return
    for p in model.get_encoder().parameters():
        p.requires_grad = False

def main():
    args = build_args()
    os.environ.setdefault("WANDB_DISABLED", "true")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        revision="refs/pr/10",
        use_safetensors=True,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=False,
        dtype="auto",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    maybe_freeze_encoder(model, args.freeze_encoder)

    ds = load_dataset(args.dataset_id)

    if args.quick_test:
        ds["train"] = ds["train"].select(range(200))
        ds["validation"] = ds["validation"].select(range(50))
        ds["test"] = ds["test"].select(range(50))

    preprocess = lambda batch: preprocess_batch(
        batch, tokenizer, args.max_input_length, args.max_target_length
    )
    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
        num_proc=args.num_proc
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)

    rouge = evaluate.load("rouge")
    compute_metrics = build_compute_metrics(tokenizer, rouge)

    eval_steps = 1500 if not args.quick_test else 100
    save_steps = eval_steps

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
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
        torch_compile=False,
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

    metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        if k.startswith("test_"):
            print(f"{k}: {v}")

    sample_idx = 0
    if len(ds["test"]) > 0:
        raw_doc = ds["test"][sample_idx]["document"]
        inputs = tokenizer([raw_doc], max_length=args.max_input_length, truncation=True, return_tensors="pt")
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
