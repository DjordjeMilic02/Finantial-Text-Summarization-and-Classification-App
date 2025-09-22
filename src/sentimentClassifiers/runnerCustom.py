import os
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HAN_DIR = Path("./scratch_han_artifacts")

PAD_IDX, UNK_IDX = 0, 1
LOWERCASE = True
NORM_NUM = True
NORM_URL = True

_re_url  = re.compile(r'https?://\S+|www\.\S+')
_re_num  = re.compile(r'\b\d+(?:\.\d+)?\b')
_re_tok  = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_re_sent = re.compile(r'(?<=[\.\!\?])\s+')

def _tok(text: str) -> List[str]:
    s = text or ""
    if LOWERCASE: s = s.lower()
    if NORM_URL:  s = _re_url.sub(" URL ", s)
    if NORM_NUM:  s = _re_num.sub(" NUM ", s)
    return _re_tok.findall(s)

def _split(text: str) -> List[str]:
    parts = _re_sent.split((text or "").strip())
    parts = [p for p in parts if p.strip()]
    return parts if parts else [text.strip()]

def _enc_tokens(tokens: List[str], stoi: Dict[str,int]) -> List[int]:
    return [stoi.get(t, UNK_IDX) for t in tokens]

def _enc_doc(text: str, stoi: Dict[str,int], max_sents: int, max_words: int) -> List[List[int]]:
    sents = _split(text)
    ids_2d = []
    for s in sents[:max_sents]:
        toks = _tok(s)
        ids = _enc_tokens(toks, stoi)
        if len(ids) < max_words:
            ids = ids + [PAD_IDX] * (max_words - len(ids))
        else:
            ids = ids[:max_words]
        ids_2d.append(ids)
    if len(ids_2d) < max_sents:
        pad_sent = [PAD_IDX] * max_words
        ids_2d += [pad_sent] * (max_sents - len(ids_2d))
    return ids_2d

class WordAttn(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, H):
        B,S,T,d = H.size()
        h = H.view(B*S, T, d)
        a = torch.tanh(self.w(h))
        e = self.u(a).squeeze(-1)
        w = torch.softmax(e, dim=1)
        s = torch.sum(h * w.unsqueeze(-1), dim=1)
        return s.view(B, S, d)

class SentAttn(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, Svec):
        a = torch.tanh(self.w(Svec))
        e = self.u(a).squeeze(-1)
        w = torch.softmax(e, dim=1)
        doc = torch.sum(Svec * w.unsqueeze(-1), dim=1)
        return doc

class HAN(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim, word_hid, sent_hid, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.emb_drop = nn.Dropout(dropout)
        self.word_rnn = nn.GRU(input_size=emb_dim, hidden_size=word_hid,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.word_attn = WordAttn(in_dim=2*word_hid, attn_dim=word_hid)
        self.sent_rnn = nn.GRU(input_size=2*word_hid, hidden_size=sent_hid,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.sent_attn = SentAttn(in_dim=2*sent_hid, attn_dim=sent_hid)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*sent_hid, num_classes)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, X):
        B,S,T = X.size()
        x = X.view(B*S, T)
        e = self.emb_drop(self.emb(x))
        w_out, _ = self.word_rnn(e)
        w_out4 = w_out.view(B, S, T, -1)
        s_vec = self.word_attn(w_out4)
        s_out, _ = self.sent_rnn(s_vec)
        doc_vec = self.sent_attn(s_out)
        doc_vec = self.dropout(doc_vec)
        logits = self.fc(doc_vec)
        return logits

_loaded = False
_device = None
_model = None
_cfg = None
_stoi = None
_id2label = None

def _lazy_load() -> None:
    global _loaded, _device, _model, _cfg, _stoi, _id2label
    if _loaded:
        return

    ckpt_path = HAN_DIR / "best.pt"
    vocab_path = HAN_DIR / "vocab.json"
    if not (ckpt_path.exists() and vocab_path.exists()):
        raise FileNotFoundError("Custom HAN artifacts not found (best.pt / vocab.json).")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = json.loads(vocab_path.read_text(encoding="utf-8"))
    _stoi = meta["stoi"]
    label2id = meta["label2id"]
    _id2label = {int(v): k for k, v in label2id.items()}

    ckpt = torch.load(ckpt_path, map_location=_device)
    cfg = ckpt.get("config", {})
    _cfg = {
        "EMB_DIM":   int(cfg.get("EMB_DIM", 128)),
        "WORD_HID":  int(cfg.get("WORD_HID", 64)),
        "SENT_HID":  int(cfg.get("SENT_HID", 64)),
        "DROPOUT":   float(cfg.get("DROPOUT", 0.3)),
        "MAX_SENTS": int(cfg.get("MAX_SENTS", 40)),
        "MAX_WORDS": int(cfg.get("MAX_WORDS", 30)),
        "VOCAB":     int(ckpt.get("vocab_size")),
        "NUM_C":     int(ckpt.get("num_classes")),
    }
    _model = HAN(
        _cfg["VOCAB"], _cfg["NUM_C"],
        _cfg["EMB_DIM"], _cfg["WORD_HID"], _cfg["SENT_HID"], _cfg["DROPOUT"]
    ).to(_device).eval()
    _model.load_state_dict(ckpt["state_dict"])

    _loaded = True

def predict(text: str) -> Tuple[str, float]:
    _lazy_load()
    if not text or not text.strip():
        return ("", 0.0)

    ids2d = _enc_doc(text, _stoi, _cfg["MAX_SENTS"], _cfg["MAX_WORDS"])
    x = torch.tensor(ids2d, dtype=torch.long).unsqueeze(0).to(_device)

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if _device.type == "cuda" else nullcontext()
    with torch.inference_mode(), amp_ctx:
        logits = _model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

    label = _id2label.get(pred_id, str(pred_id))
    return (label.capitalize(), conf)
