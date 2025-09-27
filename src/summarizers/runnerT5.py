import os
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ACCELERATE_USE_DEVICE_MAP", "0")
os.environ.setdefault("ACCELERATE_DISABLE_MMAP", "1")
hf_logging.set_verbosity_error()

_MODEL_DIR = Path(os.getenv("T5_CB_DIR", "./t5-cb-speeches"))

_tok = None
_model = None
_loaded_device = torch.device("cpu")

def _safe_max_src_len(tok) -> int:
    try:
        m = getattr(tok, "model_max_length", 1024) or 1024
        if m > 10000:
            return 1024
        return max(128, int(m))
    except Exception:
        return 1024

def _has_meta_tensors(m: torch.nn.Module) -> bool:
    for p in m.parameters():
        if getattr(p, "is_meta", False):
            return True
    for b in m.buffers():
        if getattr(b, "is_meta", False):
            return True
    return False

def _lazy_load():
    global _tok, _model, _loaded_device
    if _tok is not None and _model is not None:
        return
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model directory not found: {_MODEL_DIR}")

    _tok = AutoTokenizer.from_pretrained(_MODEL_DIR, use_fast=True, local_files_only=True)

    _model = AutoModelForSeq2SeqLM.from_pretrained(
        _MODEL_DIR,
        local_files_only=True,
        device_map=None,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    _model.to("cpu").eval()
    _loaded_device = torch.device("cpu")
    if _has_meta_tensors(_model):
        raise RuntimeError(
            "Model still on 'meta' after CPU load. Re-save checkpoint without accelerate."
        )

    if torch.cuda.is_available():
        _model.to("cuda", dtype=torch.float16)
        _loaded_device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def _chunk_ids(ids: List[int], chunk_tokens: int):
    for i in range(0, len(ids), chunk_tokens):
        yield ids[i:i + chunk_tokens]

def _summarize_chunk(input_text: str, max_new: int = 128, min_new: int = 32) -> str:
    safe_max = _safe_max_src_len(_tok)
    enc = _tok(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=safe_max,
    )
    enc = {k: v.to(_loaded_device) for k, v in enc.items()}

    with torch.inference_mode():
        if _loaded_device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = _model.generate(
                    **enc,
                    num_beams=4,
                    length_penalty=0.9,
                    max_new_tokens=max_new,
                    min_new_tokens=min_new,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
        else:
            out = _model.generate(
                **enc,
                num_beams=4,
                length_penalty=0.9,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
    return _tok.decode(out[0], skip_special_tokens=True).strip()

def summarize_cb(text: str) -> str:
    return summarize_text(text)

def summarize_t5(text: str) -> str:
    return summarize_text(text)

def summarize_text(text: str) -> str:
    _lazy_load()
    text = (text or "").strip()
    if not text:
        return ""

    prefixed = "summarize: " + text

    safe_max = _safe_max_src_len(_tok)
    chunk_tokens = max(64, safe_max - 16)

    ids = _tok.encode(prefixed, truncation=False, add_special_tokens=True)
    chunks = list(_chunk_ids(ids, chunk_tokens))

    partials: List[str] = []
    for ch in chunks:
        chunk_text = _tok.decode(ch, skip_special_tokens=True)
        partials.append(_summarize_chunk(chunk_text, max_new=128, min_new=32))

    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]

    joined = " ".join(partials)
    return _summarize_chunk("summarize: " + joined, max_new=160, min_new=50)

def summarize(text: str) -> str:
    return summarize_cb(text)
