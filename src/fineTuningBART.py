from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "kdave/Indian_Financial_News"
QUICK_TEST = False

rouge = evaluate.load("rouge")

def preprocess_data(examples, tokenizer, max_input_length=1024, max_target_length=128):
    inputs = examples["Content"]
    targets = examples["Summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]
    model_inputs["labels"] = labels

    return model_inputs

def main():
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if QUICK_TEST:
        train_dataset = train_dataset.shuffle(seed=42).select(range(len(train_dataset) // 10))
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(len(eval_dataset) // 10))
        num_train_epochs = 1
        print(f"[INFO] Quick test: {len(train_dataset)} train / {len(eval_dataset)} eval samples")
    else:
        num_train_epochs = 3
        print(f"[INFO] Full run: {len(train_dataset)} train / {len(eval_dataset)} eval samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    train_dataset = train_dataset.map(
        lambda batch: preprocess_data(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda batch: preprocess_data(batch, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = [
            [token if token != -100 else tokenizer.pad_token_id for token in label]
            for label in labels
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
        return {key: value * 100 for key, value in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bart-financial-finetuned",
        eval_strategy="steps",         
        eval_steps=500,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
        generation_max_length=128,
        generation_num_beams=4,
        predict_with_generate=True,
        fp16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("./bart-financial-finetuned-final")
    tokenizer.save_pretrained("./bart-financial-finetuned-final")

    print("Fine-tuning complete. Model saved in ./bart-financial-finetuned-final")


if __name__ == "__main__":
    main()
