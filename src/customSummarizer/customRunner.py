from __future__ import annotations
import os
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

MODEL_DIR = Path(os.getenv("CB_CUSTOM_MODEL_DIR", "runs/cb_hier_rnn_v2"))
MAX_TOKENS_FALLBACK = int(os.getenv("CB_CUSTOM_MAX_TOKENS", "140"))
MIN_SENT_LEN = 8

STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","at","by","with","from","as",
    "is","are","was","were","be","been","being","that","this","it","its","their","his","her",
    "they","them","we","you","your","i","he","she","but","if","then","than","so","such","not",
    "no","nor","do","does","did","have","has","had","about","into","over","under","between",
    "within","out","also","more","most","less","least","very","much","many"
}

_SENT_RE = re.compile(r'(?<!\b[A-Z]\.)(?<!\bU\.S)(?<=[\.\!\?])\s+')

def _split_sents(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def _tokenize_simple(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", s.lower())

def _score_sentences(sents: List[str]) -> List[Tuple[int, float]]:
    freq: Dict[str, int] = {}
    for s in sents:
        for w in _tokenize_simple(s):
            if w in STOPWORDS or len(w) <= 2:
                continue
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return [(i, 0.0) for i in range(len(sents))]
    scores = []
    for i, s in enumerate(sents):
        toks = [w for w in _tokenize_simple(s) if w not in STOPWORDS and len(w) > 2]
        if not toks:
            scores.append((i, 0.0)); continue
        s_len = max(len(toks), 1)
        score = sum(freq.get(w, 0) for w in toks) / math.sqrt(s_len)
        scores.append((i, float(score)))
    return scores

def _select_top_by_budget(sents: List[str], scores: List[Tuple[int,float]], budget_tokens: int) -> List[str]:
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    chosen = []
    used = set()
    total = 0
    for idx, _ in ranked:
        if idx in used:
            continue
        cand = sents[idx]
        if len(_tokenize_simple(cand)) < MIN_SENT_LEN:
            continue
        t = len(cand.split())
        if total + t <= budget_tokens or not chosen:
            chosen.append((idx, cand))
            total += t
            used.add(idx)
        if total >= budget_tokens:
            break
    chosen.sort(key=lambda x: x[0])
    return [c for _, c in chosen]

class SPTok:
    def __init__(self, model_path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text or "", out_type=int)

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.eos_id, self.bos_id)]
        return self.sp.decode(ids)

class WordAttn(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, H, mask):
        B,S,T,D = H.size()
        h = H.view(B*S, T, D)
        m = mask.view(B*S, T)
        a = torch.tanh(self.w(h))
        e = self.u(a).squeeze(-1)
        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~m, neg_inf)
        all_masked = ~m.any(dim=1)
        if all_masked.any():
            e[all_masked] = neg_inf
            e[all_masked, 0] = 0.0
        w = torch.softmax(e, dim=1)
        s = torch.sum(h * w.unsqueeze(-1), dim=1)
        return s.view(B, S, -1)

class HierEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, word_hid, sent_hid, pad_id, dropout=0.25, shared_emb: Optional[nn.Embedding]=None):
        super().__init__()
        self.pad_id = pad_id
        self.emb = shared_emb if shared_emb is not None else nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        self.word_rnn = nn.GRU(emb_dim, word_hid, batch_first=True, bidirectional=True)
        self.word_attn = WordAttn(2*word_hid, word_hid)
        self.sent_rnn = nn.GRU(2*word_hid, sent_hid, batch_first=True, bidirectional=True)
        self.sent_drop = nn.Dropout(dropout)
    def forward(self, enc, enc_token_mask, sent_mask):
        B,S,T = enc.size()
        x = enc.view(B*S, T)
        e = self.emb_drop(self.emb(x))
        w_out, _ = self.word_rnn(e)
        w_out = w_out.view(B, S, T, -1)
        sent_vec = self.word_attn(w_out, enc_token_mask)
        s_out, _ = self.sent_rnn(sent_vec)
        s_out = self.sent_drop(s_out)
        return s_out, sent_mask

class BahdanauAttn(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v   = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, H, s, mask):
        hs = self.W_h(H) + self.W_s(s).unsqueeze(1)
        e = self.v(torch.tanh(hs)).squeeze(-1)
        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~mask, neg_inf)
        a = torch.softmax(e, dim=1)
        ctx = torch.sum(H * a.unsqueeze(-1), dim=1)
        return ctx, a

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, dec_hid, enc_dim, pad_id, dropout=0.25, tie_embed=False, shared_emb: Optional[nn.Embedding]=None):
        super().__init__()
        self.pad_id = pad_id
        self.tie_embed = bool(tie_embed)
        self.emb = shared_emb if (self.tie_embed and shared_emb is not None) else nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn  = nn.GRU(emb_dim + enc_dim, dec_hid, batch_first=True)
        self.attn = BahdanauAttn(enc_dim, dec_hid)
        if self.tie_embed:
            self.proj = nn.Linear(dec_hid + enc_dim, emb_dim, bias=False)
            self.out_bias = nn.Parameter(torch.zeros(vocab_size))
            self.out = None
        else:
            self.out = nn.Linear(dec_hid + enc_dim, vocab_size)
    def forward(self, y_in, H_sent, sent_mask, init_h=None, return_state: bool=False):
        B,L = y_in.size()
        E = self.emb(y_in)
        logits = []
        h = init_h
        for t in range(L):
            et = E[:, t, :]
            h_t = torch.zeros(1, B, self.rnn.hidden_size, device=E.device, dtype=E.dtype) if h is None else h
            ctx, _ = self.attn(H_sent, h_t.squeeze(0), sent_mask)
            inp = torch.cat([et, ctx], dim=-1).unsqueeze(1)
            out, h = self.rnn(inp, h_t)
            o = out.squeeze(1)
            o = torch.cat([o, ctx], dim=-1)
            o = self.drop(o)
            if self.tie_embed:
                o2 = self.proj(o)
                logit = F.linear(o2, self.emb.weight, self.out_bias)
            else:
                logit = self.out(o)
            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        return (logits, h) if return_state else logits

class HierRNNSummarizer(nn.Module):
    def __init__(self, vocab_size, pad_id, emb_dim, word_hid, sent_hid, dec_hid, dropout, tie_embed: bool):
        super().__init__()
        enc_dim = 2 * sent_hid
        shared_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = HierEncoder(vocab_size, emb_dim, word_hid, sent_hid, pad_id, dropout, shared_emb=shared_emb)
        self.decoder = Decoder(vocab_size, emb_dim, dec_hid, enc_dim, pad_id, dropout, tie_embed=tie_embed, shared_emb=shared_emb if tie_embed else None)
        self.init_proj = nn.Linear(enc_dim, dec_hid)
        self.init_ln = nn.LayerNorm(enc_dim)
        self.pad_id = pad_id
    def _init_hidden(self, H_sent, sent_mask):
        m = sent_mask.float().unsqueeze(-1)
        pooled = (H_sent * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        h0 = torch.tanh(self.init_proj(self.init_ln(pooled))).unsqueeze(0)
        return h0
    def forward(self, batch):
        H_sent, sent_mask = self.encoder(batch["enc"], batch["enc_token_mask"], batch["sent_mask"])
        init_h = self._init_hidden(H_sent, sent_mask)
        logits = self.decoder(batch["y_in"], H_sent, sent_mask, init_h=init_h)
        return logits
    @torch.no_grad()
    def generate(self, enc, enc_token_mask, sent_mask, sp_tok: SPTok, max_len: int, top_k: int = 8, no_repeat_ngram_size: int = 3, min_len: int = 10):
        H_sent, s_mask = self.encoder(enc, enc_token_mask, sent_mask)
        B = enc.size(0)
        y = torch.full((B, 1), sp_tok.bos_id, dtype=torch.long, device=enc.device)
        h = self._init_hidden(H_sent, s_mask)
        outs = []
        prev_tokens = [[] for _ in range(B)]
        for t in range(max_len):
            logits, h = self.decoder(y[:, -1:], H_sent, s_mask, init_h=h, return_state=True)
            logit = logits[:, -1, :]
            next_ids = []
            for b in range(B):
                scores = logit[b]
                if top_k > 1:
                    k = min(top_k, scores.size(0))
                    cand_ids = torch.topk(scores, k=k, dim=-1).indices.tolist()
                else:
                    cand_ids = [int(torch.argmax(scores).item())]
                chosen = cand_ids[0]
                if no_repeat_ngram_size >= 3 and len(prev_tokens[b]) >= no_repeat_ngram_size - 1:
                    seen = set()
                    toks = prev_tokens[b]
                    for i in range(len(toks) - (no_repeat_ngram_size - 1)):
                        seen.add(tuple(toks[i:i+no_repeat_ngram_size]))
                    for cid in cand_ids:
                        trig = tuple((toks + [cid])[-no_repeat_ngram_size:])
                        if trig not in seen:
                            chosen = cid
                            break
                if t < min_len and chosen == sp_tok.eos_id and len(cand_ids) > 1:
                    chosen = cand_ids[1]
                next_ids.append(chosen)
            next_id = torch.tensor(next_ids, device=enc.device, dtype=torch.long).unsqueeze(-1)
            for b in range(B):
                prev_tokens[b].append(int(next_ids[b]))
            outs.append(next_id)
            y = torch.cat([y, next_id], dim=1)
            if all(next_ids[b] == sp_tok.eos_id for b in range(B)):
                break
        gen = torch.cat(outs, dim=1) if outs else torch.empty(B, 0, dtype=torch.long, device=enc.device)
        return gen

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SPTOK: Optional[SPTok] = None
_MODEL: Optional[HierRNNSummarizer] = None
_CFG: Optional[Dict] = None
_HAS_MODEL = False

def _load_latest_model() -> bool:
    global _SPTOK, _MODEL, _CFG, _HAS_MODEL
    try:
        ckpt_path = MODEL_DIR / "best.pt"
        spm_path  = MODEL_DIR / "tokenizer.model"
        if not (ckpt_path.exists() and spm_path.exists()):
            _HAS_MODEL = False
            return False
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        cfg = ckpt.get("config", {})
        vocab_size = ckpt.get("vocab_size", None)
        pad_id = ckpt.get("pad_id", 0)
        if vocab_size is None:
            for k, v in state.items():
                if k.endswith("encoder.emb.weight"):
                    vocab_size = v.size(0); break
        EMB_DIM  = cfg.get("EMB_DIM", 384)
        WORD_HID = cfg.get("WORD_HID", 384)
        SENT_HID = cfg.get("SENT_HID", 384)
        DEC_HID  = cfg.get("DEC_HID", 768)
        DROPOUT  = cfg.get("DROPOUT", 0.25)
        MAX_SENTS = cfg.get("MAX_SENTS", 50)
        MAX_WORDS = cfg.get("MAX_WORDS", 32)
        MAX_TGT   = cfg.get("MAX_TGT", 128)
        tie_embed = any(k.startswith("decoder.proj.") for k in state.keys())
        _SPTOK = SPTok(spm_path)
        _MODEL = HierRNNSummarizer(
            vocab_size=vocab_size, pad_id=pad_id,
            emb_dim=EMB_DIM, word_hid=WORD_HID, sent_hid=SENT_HID,
            dec_hid=DEC_HID, dropout=DROPOUT, tie_embed=tie_embed
        )
        _MODEL.load_state_dict(state, strict=True)
        _MODEL.eval().to(_DEVICE)
        _CFG = {"MAX_SENTS": MAX_SENTS, "MAX_WORDS": MAX_WORDS, "MAX_TGT": MAX_TGT}
        _HAS_MODEL = True
        return True
    except Exception as e:
        _HAS_MODEL = False
        _MODEL = None
        _SPTOK = None
        _CFG = None
        return False

def _ensure_loaded() -> bool:
    global _HAS_MODEL
    if _HAS_MODEL and (_MODEL is not None) and (_SPTOK is not None):
        return True
    return _load_latest_model()

def _encode_document_for_model(text: str):
    sents = [s for s in _split_sents(text) if s]
    if not sents:
        sents = [text.strip()]
    max_sents = max(1, _CFG["MAX_SENTS"])
    max_words = max(1, _CFG["MAX_WORDS"])
    sents = sents[:max_sents]
    tokenized = []
    for s in sents:
        ids = _SPTOK.encode(s)[:max_words]
        if not ids:
            ids = [_SPTOK.bos_id, _SPTOK.eos_id]
        tokenized.append(ids)
    T = max(len(t) for t in tokenized)
    enc = []
    mask_tok = []
    for ids in tokenized:
        l = len(ids)
        enc.append(ids + [_SPTOK.pad_id]*(T - l))
        mask_tok.append([True]*l + [False]*(T - l))
    enc = torch.tensor([enc], dtype=torch.long, device=_DEVICE)
    enc_token_mask = torch.tensor([mask_tok], dtype=torch.bool, device=_DEVICE)
    sent_mask = torch.tensor([[True]*len(tokenized)], dtype=torch.bool, device=_DEVICE)
    return enc, enc_token_mask, sent_mask

def _summarize_with_model(text: str) -> str:
    enc, enc_token_mask, sent_mask = _encode_document_for_model(text)
    gen_ids = _MODEL.generate(
        enc, enc_token_mask, sent_mask, _SPTOK,
        max_len=_CFG["MAX_TGT"], top_k=8, no_repeat_ngram_size=3, min_len=10
    )
    return _SPTOK.decode(gen_ids[0].tolist())

def summarize_cb_custom(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if _ensure_loaded():
        try:
            return _summarize_with_model(text)
        except Exception:
            pass
    sents = _split_sents(text)
    if not sents:
        return ""
    scores = _score_sentences(sents)
    picked = _select_top_by_budget(sents, scores, budget_tokens=MAX_TOKENS_FALLBACK)
    return " ".join(picked)

def summarize_cb(text: str) -> str:
    return summarize_cb_custom(text)

def summarize(text: str) -> str:
    return summarize_cb_custom(text)
