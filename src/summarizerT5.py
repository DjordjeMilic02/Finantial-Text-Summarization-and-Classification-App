from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
import torch

def summarize_text():
    model_dir = Path("./t5-cb-speeches")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {model_dir}\n"
            "Run training or point model_dir to your saved checkpoint."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    in_path = Path("input/input.txt")
    out_path = Path("output/summaryT5.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8").strip()

    max_src_tokens = min(getattr(tokenizer, "model_max_length", 1024), 1024)
    chunk_tokens = max_src_tokens - 16

    def to_chunks(s: str):
        ids = tokenizer.encode("summarize: " + s, truncation=False, add_special_tokens=True)
        for i in range(0, len(ids), chunk_tokens):
            yield ids[i:i+chunk_tokens]

    chunks = list(to_chunks(text))

    partial_summaries = []
    for ids in chunks:
        chunk_text = tokenizer.decode(ids, skip_special_tokens=True)
        out = summarizer(
            chunk_text,
            max_length=128,
            min_length=32,
            do_sample=False
        )[0]["summary_text"]
        partial_summaries.append(out.strip())

    if len(partial_summaries) > 1:
        joined = " ".join(partial_summaries)
        final_in = "summarize: " + joined
        final = summarizer(
            final_in,
            max_length=160,
            min_length=50,
            do_sample=False
        )[0]["summary_text"].strip()
        summary = final
    else:
        summary = partial_summaries[0] if partial_summaries else ""

    out_path.write_text(summary, encoding="utf-8")
    print(f"T5 summary written to {out_path}")

if __name__ == "__main__":
    summarize_text()
