import os, warnings
from typing import Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    logging as hf_logging,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_DIR  = "./cb-stance-flare"
MAX_LENGTH = 512
USE_FP16   = True

_tok = None
_model = None
_device = None

def _load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()
    return tok, model, device

def predict(text: str) -> Tuple[str, float]:
    global _tok, _model, _device
    if _tok is None:
        _tok, _model, _device = _load_model()

    enc = _tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.inference_mode():
        if USE_FP16 and _device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = _model(**enc).logits
        else:
            logits = _model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred_id].item())

    label = _model.config.id2label.get(pred_id, str(pred_id))
    return label, conf
