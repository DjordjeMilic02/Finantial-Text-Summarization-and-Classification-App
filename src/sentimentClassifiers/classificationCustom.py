import os
import json
import re
import argparse
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

PAD_IDX, UNK_IDX = 0, 1
LOWERCASE = True
NORM_NUM = True
NORM_URL = True

_re_url = re.compile(r'https?://\S+|www\.\S+')
_re_num = re.compile(r'\b\d+(?:\.\d+)?\b')
_re_tok = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_re_sent = re.compile(r'(?<=[\.\!\?])\s+')

def tokenize(text: str) -> List[str]:
    s = text or ""
    if LOWERCASE: s = s.lower()
    if NORM_URL:  s = _re_url.sub(" URL ", s)
    if NORM_NUM:  s = _re_num.sub(" NUM ", s)
    return _re_tok.findall(s)

def split_sentences(text: str) -> List[str]:
    parts = _re_sent.split((text or "").strip())
    parts = [p for p in parts if p.strip()]
    return parts if parts else [text.strip()]

def encode_tokens(tokens: List[str], stoi: Dict[str,int]) -> List[int]:
    return [stoi.get(t, UNK_IDX) for t in tokens]

def encode_doc(text: str, stoi: Dict[str,int], max_sents: int, max_words: int) -> List[List[int]]:
    sents = split_sentences(text)
    ids_2d = []
    for s in sents[:max_sents]:
        toks = tokenize(s)
        ids = encode_tokens(toks, stoi)
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

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="scratch_han_artifacts",
                    help="Folder with best.pt and vocab.json")
    ap.add_argument("--input_txt", type=str, default="input/testInputBank.txt",
                    help="Path to a UTF-8 text file to classify")
    ap.add_argument("--fp16", action="store_true", help="Use CUDA fp16 for inference")
    return ap.parse_args()

def main():
    args = build_args()
    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / "best.pt"
    vocab_path = model_dir / "vocab.json"

    if not ckpt_path.exists() or not vocab_path.exists():
        print("error: missing best.pt or vocab.json in model_dir")
        return

    try:
        text = Path(args.input_txt).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print("error: input file not found")
        return
    if not text:
        print("error: empty input")
        return

    meta = json.loads(vocab_path.read_text(encoding="utf-8"))
    stoi = meta["stoi"]
    label2id = meta["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    vocab_size = int(ckpt.get("vocab_size"))
    num_classes = int(ckpt.get("num_classes"))
    EMB_DIM = int(cfg.get("EMB_DIM", 128))
    WORD_HID = int(cfg.get("WORD_HID", 64))
    SENT_HID = int(cfg.get("SENT_HID", 64))
    DROPOUT = float(cfg.get("DROPOUT", 0.3))
    MAX_SENTS = int(cfg.get("MAX_SENTS", 40))
    MAX_WORDS = int(cfg.get("MAX_WORDS", 30))

    model = HAN(vocab_size, num_classes, EMB_DIM, WORD_HID, SENT_HID, DROPOUT).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    ids2d = encode_doc(text, stoi, MAX_SENTS, MAX_WORDS)
    x = torch.tensor(ids2d, dtype=torch.long).unsqueeze(0).to(device)

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (args.fp16 and device.type == "cuda")
        else nullcontext()
    )
    with torch.inference_mode(), amp_ctx:
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())

    label = id2label.get(pred_id, str(pred_id))
    print(f"{label}\t{conf:.4f}")

if __name__ == "__main__":
    main()
