import os
from pathlib import Path
from typing import Optional, List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging as hf_logging

MODEL_DIR = Path("./bart-financial-finetuned-final")
BASE_ID = "facebook/bart-large-cnn"

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

def _load_bart():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_ID)
    return tok, model

def _chunk_ids(ids: List[int], chunk_len: int) -> List[List[int]]:
    return [ids[i:i+chunk_len] for i in range(0, len(ids), chunk_len)]

def summarize_news(text: str) -> str:
    tok, model = _load_bart()
    pipe = pipeline("summarization", model=model, tokenizer=tok, device_map="auto")

    max_src = min(getattr(tok, "model_max_length", 1024), 1024)
    chunk_len = max_src - 32

    ids = tok.encode(text, truncation=False)
    chunks = _chunk_ids(ids, chunk_len)
    parts = []

    for ch in chunks:
        ch_txt = tok.decode(ch, skip_special_tokens=True)
        out = pipe(ch_txt, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]
        parts.append(out.strip())

    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]

    joined = " ".join(parts)
    final = pipe(joined, max_length=220, min_length=80, do_sample=False)[0]["summary_text"]
    return final.strip()

if __name__ == "__main__":
    sample = Path("input/input.txt").read_text(encoding="utf-8") if Path("input/input.txt").exists() else \
             "Error!"
    print(summarize_news(sample))
