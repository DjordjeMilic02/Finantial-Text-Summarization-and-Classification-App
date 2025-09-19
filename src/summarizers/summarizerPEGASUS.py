from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import sys
from pathlib import Path

def summarize_text():

    model_dir = Path("./pegasus-earnings-fast")
    base_model_id = "google/pegasus-large"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {model_dir}\n"
            f"Make sure it matches your training script's --output_dir."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        use_safetensors=True,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True
    )

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    in_path = Path("input/input.txt")
    out_path = Path("output/summaryPEGASUS.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8")

    max_tokens = getattr(tokenizer, "model_max_length", 1024)
    if len(tokenizer.encode(text)) > max_tokens:
        ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        text = tokenizer.decode(ids, skip_special_tokens=True)

    summary = summarizer(
        text,
        max_length=150,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]

    out_path.write_text(summary, encoding="utf-8")
    print(f"Pegasus summary written to {out_path}")

if __name__ == "__main__":
    try:
        summarize_text()
    except OSError as e:
        msg = str(e)
        if "safetensors" in msg.lower() or "pytorch_model.bin" in msg.lower():
            print(
                "\n[Hint] Loading failed. Ensure your fine-tuned folder contains 'model.safetensors'.\n"
                "If it only has 'pytorch_model.bin' and you cannot upgrade torch,\n"
                "re-run training with safetensors saving enabled (or re-save the model with safe serialization).",
                file=sys.stderr
            )
        raise
