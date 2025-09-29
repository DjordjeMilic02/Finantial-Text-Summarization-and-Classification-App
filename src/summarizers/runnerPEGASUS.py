import os
import re
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers.utils import logging as hf_logging

MODEL_DIR = Path("./pegasus-earnings-fast")
BASE_ID   = "google/pegasus-large"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
hf_logging.set_verbosity_error()

_boilerplate_rx = re.compile(
    r"(forward[- ]looking statements?[^.]*\.)|"
    r"(safe harbor[^.]*\.)|"
    r"(private securities litigation reform act[^.]*\.)|"
    r"(Form\s+10[-KQ]\b)|(\bForm\s+8[-K]\b)|"
    r"(\bSEC(urities and Exchange Commission)?\b)|"
    r"(copyright\s+[\d\-–]+)|(all rights reserved)",
    flags=re.IGNORECASE
)

def _strip_boilerplate(text: str) -> str:
    return _boilerplate_rx.sub("", text)

def _load_pegasus():
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        use_safetensors=True,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    return tok, model

def _chunk_ids_overlap(ids: List[int], chunk_len: int, stride: int) -> List[List[int]]:
    """Overlapping windows: step = chunk_len - stride."""
    step = max(1, chunk_len - stride)
    out = []
    for i in range(0, len(ids), step):
        win = ids[i : i + chunk_len]
        if not win:
            break
        out.append(win)
        if i + chunk_len >= len(ids):
            break
    return out

def _truncate_to_tokens(text: str, tok, max_tokens: int) -> str:
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    return tok.decode(ids, skip_special_tokens=True)

def summarize_earnings(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _strip_boilerplate(text)

    tok, model = _load_pegasus()
    pipe = pipeline("summarization", model=model, tokenizer=tok, device_map="auto")

    SRC_LEN = 768
    STRIDE  = 96

    DECODE_KW = dict(
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=0.95,
        early_stopping=True,
        do_sample=False,
    )

    ids = tok.encode(text, truncation=False)
    if len(ids) <= SRC_LEN:
        chunk_txt = _truncate_to_tokens(text, tok, SRC_LEN)
        out = pipe(chunk_txt, max_new_tokens=110, min_length=30, **DECODE_KW)[0]["summary_text"]
        return out.strip()

    chunks = _chunk_ids_overlap(ids, chunk_len=SRC_LEN, stride=STRIDE)
    parts: List[str] = []
    for ch in chunks:
        ch_txt = tok.decode(ch, skip_special_tokens=True)
        part = pipe(ch_txt, max_new_tokens=110, min_length=30, **DECODE_KW)[0]["summary_text"]
        parts.append(part.strip())

    if not parts:
        return ""

    stitched = " ".join(parts)
    stitched = _truncate_to_tokens(stitched, tok, SRC_LEN)
    final = pipe(stitched, max_new_tokens=140, min_length=60, **DECODE_KW)[0]["summary_text"]
    return final.strip()

if __name__ == "__main__":
    in_path = Path("input/input.txt")
    sample = in_path.read_text(encoding="utf-8").strip() if in_path.exists() else ""
    print(summarize_earnings(sample))
