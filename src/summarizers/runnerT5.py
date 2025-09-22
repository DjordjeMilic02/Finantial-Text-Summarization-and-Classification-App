import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging as hf_logging

MODEL_DIR = Path("./t5-cb-speeches")

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

def _load_t5():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("summarization", model=model, tokenizer=tok, device=device)
    return tok, pipe

def _chunk_ids(ids: List[int], chunk_len: int) -> List[List[int]]:
    return [ids[i:i+chunk_len] for i in range(0, len(ids), chunk_len)]

def summarize_cb(text: str) -> str:
    tok, pipe = _load_t5()

    src = "summarize: " + text.strip()

    max_src = min(getattr(tok, "model_max_length", 1024), 1024)
    chunk_len = max_src - 16

    ids = tok.encode(src, truncation=False, add_special_tokens=True)
    chunks = _chunk_ids(ids, chunk_len)

    parts = []
    for ch in chunks:
        ch_txt = tok.decode(ch, skip_special_tokens=True)
        out = pipe(ch_txt, max_length=128, min_length=32, do_sample=False)[0]["summary_text"]
        parts.append(out.strip())

    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]

    joined = "summarize: " + " ".join(parts)
    final = pipe(joined, max_length=160, min_length=50, do_sample=False)[0]["summary_text"].strip()
    return final

if __name__ == "__main__":
    sample = Path("input/input.txt").read_text(encoding="utf-8") if Path("input/input.txt").exists() else \
             "Error!"
    print(summarize_cb(sample))
