import os
import sys
import json
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    logging as hf_logging,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()

_DEFAULT_DIR = os.environ.get("NEWS_MODEL_DIR", "./fullnews-longformer-opendatabay")
_DEFAULT_MAX_LEN = int(os.environ.get("NEWS_MAX_LEN", "2048"))
_DEFAULT_USE_GPU = os.environ.get("NEWS_USE_GPU", "0") in {"1", "true", "True"}

_STATE = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "model_dir": None,
    "model_type": None,
    "max_length": _DEFAULT_MAX_LEN,
    "use_gpu": _DEFAULT_USE_GPU,
}

def _retry(fn, retries: int = 3, delay: float = 0.25):
    last = None
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last

def _load_state_single(path: Path):
    if path.suffix == ".bin":
        return torch.load(path, map_location="cpu")
    elif path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    raise FileNotFoundError(str(path))

def _load_state_sharded(folder: Path, index_json: Path):
    with index_json.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map") or {}
    shards = sorted(set(weight_map.values()))
    state = {}
    for shard in shards:
        shard_path = folder / shard
        part = _load_state_single(shard_path)
        state.update(part)
    return state

def _manual_build_and_load(model_dir: Path):
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_config(cfg)
    bin_single = model_dir / "pytorch_model.bin"
    safetensors_single = model_dir / "model.safetensors"
    bin_index = model_dir / "pytorch_model.bin.index.json"
    safe_index = model_dir / "model.safetensors.index.json"

    if bin_index.exists():
        state = _load_state_sharded(model_dir, bin_index)
    elif safe_index.exists():
        state = _load_state_sharded(model_dir, safe_index)
    elif safetensors_single.exists():
        state = _load_state_single(safetensors_single)
    elif bin_single.exists():
        state = _load_state_single(bin_single)
    else:
        raise FileNotFoundError("No model weights found in directory.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    return model

def _ensure_loaded(model_dir: Optional[str] = None, max_length: Optional[int] = None, use_gpu: Optional[bool] = None):
    md = model_dir or _DEFAULT_DIR
    if _STATE["model"] is not None and _STATE["model_dir"] == md:
        if max_length is not None:
            _STATE["max_length"] = int(max_length)
        if use_gpu is not None:
            _STATE["use_gpu"] = bool(use_gpu)
        return

    want_gpu = bool(_DEFAULT_USE_GPU if use_gpu is None else use_gpu)
    device = torch.device("cuda" if (want_gpu and torch.cuda.is_available()) else "cpu")

    tok = _retry(lambda: AutoTokenizer.from_pretrained(md, use_fast=True))

    def _try_normal_load():
        return AutoModelForSequenceClassification.from_pretrained(
            md,
            device_map=None,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float32,
        )

    try:
        model = _retry(_try_normal_load)
        if any(p.is_meta for p in model.parameters()):
            raise RuntimeError("meta_detected")
    except Exception:
        model = _manual_build_and_load(Path(md))

    model = model.to(device).eval()

    _STATE.update({
        "tokenizer": tok,
        "model": model,
        "device": device,
        "model_dir": md,
        "model_type": getattr(model.config, "model_type", None),
        "max_length": int(max_length) if max_length is not None else _DEFAULT_MAX_LEN,
        "use_gpu": want_gpu,
    })

def predict(text: str,
            model_dir: Optional[str] = None,
            max_length: Optional[int] = None,
            use_gpu: Optional[bool] = None) -> Tuple[str, float]:
    if not (text or "").strip():
        return "", 0.0

    _ensure_loaded(model_dir, max_length, use_gpu)

    tok = _STATE["tokenizer"]
    model = _STATE["model"]
    device = _STATE["device"]
    max_len = int(_STATE["max_length"])
    model_type = _STATE["model_type"]

    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    extra = {}
    if model_type == "longformer":
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        extra["global_attention_mask"] = global_attention_mask

    use_amp = (device.type == "cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    with torch.inference_mode(), amp_ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask, **extra)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

    id2label = getattr(model.config, "id2label", {}) or {}
    label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))
    return str(label), conf

def _read_text(path: Optional[str]) -> str:
    if path:
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
        return ""
    return (sys.stdin.read() or "").strip()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=_DEFAULT_DIR)
    ap.add_argument("--input_txt", type=str, default="")
    ap.add_argument("--max_length", type=int, default=_DEFAULT_MAX_LEN)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    text = _read_text(args.input_txt)
    if not text:
        print("empty")
        return

    label, conf = predict(text, model_dir=args.model_dir, max_length=args.max_length, use_gpu=args.gpu)
    print(f"{label}\t{conf:.4f}")

if __name__ == "__main__":
    main()
