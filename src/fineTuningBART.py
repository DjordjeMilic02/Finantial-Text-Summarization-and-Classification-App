from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

QUICK_TEST = True

MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "kdave/Indian_Financial_News"

def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=128):
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
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    dataset = load_dataset(DATASET_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    tokenized_datasets = dataset.map(
        lambda batch: preprocess_data(batch, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    if QUICK_TEST:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
        eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200, 400))
        num_train_epochs = 1
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)["test"]
        num_train_epochs = 3

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bart-financial-finetuned",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
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
        data_collator=data_collator
    )

    trainer.train()

    trainer.save_model("./bart-financial-finetuned-final")
    tokenizer.save_pretrained("./bart-financial-finetuned-final")

    print("✅ Fine-tuning complete. Model saved in ./bart-financial-finetuned-final")

if __name__ == "__main__":
    main()
