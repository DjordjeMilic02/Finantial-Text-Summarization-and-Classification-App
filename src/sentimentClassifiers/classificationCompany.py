import os
import warnings
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

MODEL_DIR  = "./earnings-aiera-finbert"
INPUT_TXT  = "input/testInputCompany.txt"
MAX_LENGTH = 256
USE_FP16   = True

def main():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("empty")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc = {k: v.to(device) for k, v in enc.items()}

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if (USE_FP16 and device.type=="cuda") else nullcontext()
    with torch.inference_mode(), amp_ctx:
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred_id].item())

    label = model.config.id2label.get(pred_id, str(pred_id))
    print(f"{label}\t{conf:.4f}")

if __name__ == "__main__":
    main()
