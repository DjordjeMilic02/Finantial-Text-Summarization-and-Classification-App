import os
import warnings
from contextlib import nullcontext
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

MODEL_DIR  = "./fullnews-longformer-opendatabay"
MAX_LENGTH = 2048
USE_FP16   = True

_tok = None
_model = None
_device = None

def _lazy_load():
    global _tok, _model, _device
    if _tok is not None:
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    hf_logging.set_verbosity_error()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device).eval()

def predict(text: str) -> Tuple[str, float]:
    _lazy_load()
    if not text or not text.strip():
        return ("", 0.0)

    enc = _tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)

    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (USE_FP16 and _device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        out = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        probs = torch.softmax(out.logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred_id].item())

    label = _model.config.id2label.get(pred_id, str(pred_id))
    label = label.capitalize()
    return (label, conf)
