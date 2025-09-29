import os
import re
import sys
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def strip_boilerplate(text: str) -> str:
    patterns = [
        r"forward[- ]looking statements?[^.]*\.",
        r"safe harbor[^.]*\.",
        r"private securities litigation reform act[^.]*\.",
        r"Form\s+10[-KQ]\b", r"\bForm\s+8[-K]\b",
        r"\bSEC(urities and Exchange Commission)?\b",
        r"copyright\s+[\d\-–]+", r"all rights reserved",
    ]
    rx = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return rx.sub("", text)


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Hard truncate a string to at most `max_tokens` using the tokenizer."""
    ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids, skip_special_tokens=True)


def chunk_by_tokens(text: str, tokenizer, max_src_tokens: int = 768, stride: int = 96) -> List[str]:
    enc = tokenizer(text, truncation=False, return_attention_mask=False)
    ids = enc["input_ids"]
    chunks = []
    step = max(1, max_src_tokens - max(0, stride))
    for i in range(0, len(ids), step):
        window = ids[i: i + max_src_tokens]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if i + max_src_tokens >= len(ids):
            break
    return chunks


def summarize_one(summarizer, text: str, max_new_tokens: int) -> str:
    """One-pass summary with anti-repetition decoding settings."""
    out = summarizer(
        text,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=0.95,
        early_stopping=True,
        max_new_tokens=max_new_tokens,
        min_length=30,
        do_sample=False,
    )[0]["summary_text"]
    return out.strip()


def summarize_map_reduce(
    text: str,
    summarizer,
    tokenizer,
    src_len: int = 768,
    chunk_stride: int = 96,
    per_chunk_new_tokens: int = 110,
    final_new_tokens: int = 140,
) -> str:
    text = strip_boilerplate(text)

    pieces = chunk_by_tokens(text, tokenizer, max_src_tokens=src_len, stride=chunk_stride)

    if len(pieces) <= 1:
        single = truncate_to_tokens(text, tokenizer, src_len)
        return summarize_one(summarizer, single, max_new_tokens=per_chunk_new_tokens)

    partials = []
    for p in pieces:
        partial = summarize_one(summarizer, p, max_new_tokens=per_chunk_new_tokens)
        partials.append(partial)

    stitched = " ".join(partials)
    stitched = truncate_to_tokens(stitched, tokenizer, src_len)
    final = summarize_one(summarizer, stitched, max_new_tokens=final_new_tokens)
    return final

def summarize_text():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
        trust_remote_code=True,
    )

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    in_path = Path("input/testInputCompany.txt")
    out_path = Path("output/summaryPEGASUS.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    text = in_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Input file is empty.")

    final_summary = summarize_map_reduce(
        text,
        summarizer=summarizer,
        tokenizer=tokenizer,
        src_len=768,
        chunk_stride=96,
        per_chunk_new_tokens=110,
        final_new_tokens=140,
    )

    out_path.write_text(final_summary, encoding="utf-8")
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
                file=sys.stderr,
            )
        raise
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
