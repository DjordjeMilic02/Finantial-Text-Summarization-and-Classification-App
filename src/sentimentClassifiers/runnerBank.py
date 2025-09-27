import os
import warnings
from typing import Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ACCELERATE_USE_DEVICE_MAP", "0")
os.environ.setdefault("ACCELERATE_DISABLE_MMAP", "1")
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_DIR  = os.getenv("CB_BANK_SENT_MODEL_DIR", "./cb-stance-flare")
MAX_LENGTH = int(os.getenv("CB_SENT_MAXLEN", "512"))
USE_FP16   = os.getenv("CB_SENT_FP16", "1") == "1"

_tok = None
_model = None
_device = None
_dtype = None

def _load_model():
    global _tok, _model, _device, _dtype
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _dtype = torch.float16 if (_device.type == "cuda" and USE_FP16) else torch.float32

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    _tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map=None,
        low_cpu_mem_usage=False,
        torch_dtype=_dtype,
    )
    _model.to(_device).eval()

def predict(text: str) -> Tuple[str, float]:
    global _tok, _model, _device
    if _tok is None or _model is None:
        _load_model()

    enc = _tok(text or "", return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.inference_mode():
        if _device.type == "cuda" and _model.dtype == torch.float16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = _model(**enc).logits
        else:
            logits = _model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred_id].item())

    label = _model.config.id2label.get(pred_id, str(pred_id))
    return label, conf
