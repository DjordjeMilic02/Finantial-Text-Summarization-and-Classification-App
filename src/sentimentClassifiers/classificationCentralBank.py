import os
import argparse
import warnings
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="./cb-stance-flare",
                    help="Folder with the fine-tuned DeBERTa (trainer output).")
    ap.add_argument("--input_txt", type=str, default="input/testInputBank.txt",
                    help="Path to the input text file.")
    ap.add_argument("--max_length", type=int, default=512,
                    help="Truncation length (should match training).")
    ap.add_argument("--fp16", action="store_true",
                    help="Use fp16 on CUDA during inference.")
    return ap.parse_args()

def main():
    args = build_args()

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    try:
        with open(args.input_txt, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print("error: input file not found")
        return
    if not text:
        print("error: empty input")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
    enc = {k: v.to(device) for k, v in enc.items()}

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (args.fp16 and device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)
        pred_id = int(probs.argmax(dim=-1).item())
        conf = float(probs[0, pred_id].item())

    label = model.config.id2label.get(pred_id, str(pred_id))
    print(f"{label}\t{conf:.4f}")

if __name__ == "__main__":
    main()
