import os
import re
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import pipeline
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

START_MODEL_PATH = "./finbert-finetuned"
LABEL_MAP = {
    0: "News",
    1: "Earnings Call",
    2: "Central Bank Speech"
}
_LABEL_RE = re.compile(r"label[_\-:\s]*([0-9]+)", re.I)

@dataclass
class StartPrediction:
    class_id: int
    class_name: str
    confidence: float

class StartClassifierRunner:
    _clf = None

    @classmethod
    def _get_pipeline(cls):
        if cls._clf is None:
            device = 0 if torch.cuda.is_available() else -1
            cls._clf = pipeline(
                "text-classification",
                model=START_MODEL_PATH,
                tokenizer=START_MODEL_PATH,
                device=device
            )
        return cls._clf

    @staticmethod
    def _label_to_id(label_str: str) -> int:
        if label_str.isdigit():
            return int(label_str)
        m = _LABEL_RE.search(label_str)
        return int(m.group(1)) if m else -1

    @classmethod
    def predict(cls, text: str, max_length: int = 512) -> StartPrediction:
        if not text or not text.strip():
            raise ValueError("Empty text passed to start classifier.")
        clf = cls._get_pipeline()
        out = clf(text, truncation=True, max_length=max_length)[0]

        label_id = StartClassifierRunner._label_to_id(str(out["label"]))
        class_name = LABEL_MAP.get(label_id, str(out["label"]))
        confidence = float(out.get("score", 0.0))

        return StartPrediction(class_id=label_id, class_name=class_name, confidence=confidence)