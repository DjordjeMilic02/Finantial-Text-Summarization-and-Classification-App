import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import sentencepiece as spm

MODEL_DIR  = "runs/cb_hier_rnn/cb_hier_rnn_v2"
CKPT_NAME  = "best.pt"
TOK_NAMES  = ["tokenizer.model", "spm.model"]
INPUT_PATH = "input/testInputBank.txt"

def split_sentences(text: str) -> List[str]:
    return [p.strip() for p in re.compile(r'(?<=[\.\!\?])\s+').split((text or "").strip()) if p.strip()]


class SPTok:
    def __init__(self, model_path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.decode(ids)

class WordAttn(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, mask):
        B, S, T, D = H.size()
        h = H.view(B * S, T, D)
        m = mask.view(B * S, T)
        a = torch.tanh(self.w(h))
        e = self.u(a).squeeze(-1)
        e = e.masked_fill(~m, torch.finfo(e.dtype).min)
        w = torch.softmax(e, dim=1)
        s = torch.sum(h * w.unsqueeze(-1), dim=1)
        return s.view(B, S, D)


class HierEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, word_hid, sent_hid, pad_id, dropout=0.2, shared_emb: Optional[nn.Embedding] = None):
        super().__init__()
        self.pad_id = pad_id
        self.emb = shared_emb if shared_emb is not None else nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        self.word_rnn = nn.GRU(emb_dim, word_hid, batch_first=True, bidirectional=True)
        self.word_attn = WordAttn(2 * word_hid, word_hid)
        self.sent_rnn = nn.GRU(2 * word_hid, sent_hid, batch_first=True, bidirectional=True)
        self.sent_drop = nn.Dropout(dropout)

    def forward(self, enc, enc_token_mask, sent_mask):
        B, S, T = enc.size()
        x = enc.view(B * S, T)
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
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, s, mask):
        hs = self.W_h(H) + self.W_s(s).unsqueeze(1)
        e = self.v(torch.tanh(hs)).squeeze(-1)
        e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
        a = torch.softmax(e, dim=1)
        ctx = torch.sum(H * a.unsqueeze(-1), dim=1)
        return ctx, a


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, dec_hid, enc_dim, pad_id, dropout=0.2,
                 shared_emb: Optional[nn.Embedding] = None, two_step: bool = False, tie_embed: bool = True):
        super().__init__()
        self.pad_id = pad_id
        self.two_step = two_step

        self.emb = shared_emb if shared_emb is not None else nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim + enc_dim, dec_hid, batch_first=True)
        self.attn = BahdanauAttn(enc_dim, dec_hid)

        if self.two_step:
            self.proj = nn.Linear(dec_hid + enc_dim, emb_dim, bias=False)
            self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.out = nn.Linear(dec_hid + enc_dim, vocab_size)
            if tie_embed and (self.emb.weight.shape == self.out.weight.shape):
                self.out.weight = self.emb.weight

    def forward(self, y_in, H_sent, sent_mask, init_h=None, return_state: bool = False):
        B, L = y_in.size()
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

            if self.two_step:
                z = self.proj(o)
                logit = torch.matmul(z, self.emb.weight.t())
                logit = logit + self.out_bias
            else:
                logit = self.out(o)

            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        if return_state:
            return logits, h
        return logits


class HierRNNSummarizer(nn.Module):
    def __init__(self, vocab_size, pad_id, emb_dim, word_hid, sent_hid, dec_hid, dropout=0.2,
                 use_init_proj: bool = True, two_step: bool = False):
        super().__init__()
        shared_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = HierEncoder(vocab_size, emb_dim, word_hid, sent_hid, pad_id, dropout=dropout, shared_emb=shared_emb)
        enc_dim = 2 * sent_hid
        self.decoder = Decoder(
            vocab_size, emb_dim, dec_hid, enc_dim, pad_id,
            dropout=dropout, shared_emb=shared_emb, two_step=two_step, tie_embed=True
        )
        self.init_proj = nn.Linear(enc_dim, dec_hid) if use_init_proj else None

    def _init_hidden(self, H_sent, sent_mask):
        if self.init_proj is None:
            return None
        m = sent_mask.float().unsqueeze(-1)
        pooled = (H_sent * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        h0 = torch.tanh(self.init_proj(pooled)).unsqueeze(0)
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
                    topk = torch.topk(scores, k=min(top_k, scores.size(0)), dim=-1)
                    cand_ids = topk.indices.tolist()
                else:
                    cand_ids = [int(torch.argmax(scores).item())]

                chosen = cand_ids[0]
                if no_repeat_ngram_size >= 3 and len(prev_tokens[b]) >= no_repeat_ngram_size - 1:
                    seen = set()
                    toks = prev_tokens[b]
                    for i in range(len(toks) - (no_repeat_ngram_size - 1)):
                        seen.add(tuple(toks[i:i + no_repeat_ngram_size]))
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

def _find_tokenizer(model_dir: Path) -> Path:
    for name in TOK_NAMES:
        p = model_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No tokenizer model found in {model_dir} (tried {TOK_NAMES})")


def _build_enc_from_text(text: str, tok: SPTok, max_sents: int, max_words: int, device: torch.device):
    sents = split_sentences(text)[:max_sents]
    sent_ids, sent_lens = [], []
    for s in sents:
        ids = tok.encode(s)[:max_words]
        sent_lens.append(len(ids))
        if len(ids) < max_words:
            ids = ids + [tok.pad_id] * (max_words - len(ids))
        sent_ids.append(ids)
    if len(sent_ids) < max_sents:
        pad_sent = [tok.pad_id] * max_words
        sent_ids += [pad_sent] * (max_sents - len(sent_ids))
        sent_lens += [0] * (max_sents - len(sent_lens))

    enc = torch.tensor([sent_ids], dtype=torch.long, device=device)
    enc_token_mask = (enc != tok.pad_id)
    sent_mask = (torch.tensor([sent_lens], dtype=torch.long, device=device) > 0)
    return enc, enc_token_mask, sent_mask


def summarize(text: str, model_dir: str) -> str:
    model_dir = Path(model_dir)
    ckpt_path = model_dir / CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    tok_path = _find_tokenizer(model_dir)
    sp_tok = SPTok(tok_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", {})
    if not sd:
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            sd = ckpt
        else:
            raise ValueError("Checkpoint missing 'state_dict'.")

    is_v2_like = any(k.startswith("init_proj.") for k in sd.keys()) or \
                 any(k.startswith("decoder.proj.") for k in sd.keys()) or \
                 any("decoder.out_bias" in k for k in sd.keys())
    if not is_v2_like:
        raise ValueError("Wrong model")

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    vocab_size = int(ckpt.get("vocab_size", sp_tok.sp.vocab_size()))
    pad_id     = int(ckpt.get("pad_id", sp_tok.pad_id))

    emb_dim   = int(cfg.get("EMB_DIM", 384))
    word_hid  = int(cfg.get("WORD_HID", 384))
    sent_hid  = int(cfg.get("SENT_HID", 384))
    dec_hid   = int(cfg.get("DEC_HID", 768))
    dropout   = float(cfg.get("DROPOUT", 0.25))
    max_sents = int(cfg.get("MAX_SENTS", 50))
    max_words = int(cfg.get("MAX_WORDS", 32))
    max_tgt   = int(cfg.get("MAX_TGT", 128))

    two_step = False
    if ("decoder.proj.weight" in sd) or ("decoder.out_bias" in sd):
        two_step = True
    elif "decoder.out.weight" in sd:
        ow = sd["decoder.out.weight"]
        if isinstance(ow, torch.Tensor) and ow.dim() == 2 and ow.size(0) != vocab_size:
            two_step = True

    if two_step and "decoder.out.weight" in sd and "decoder.proj.weight" not in sd:
        sd["decoder.proj.weight"] = sd.pop("decoder.out.weight")
    if two_step and "decoder.out.bias" in sd and "decoder.out_bias" not in sd:
        sd["decoder.out_bias"] = sd.pop("decoder.out.bias")

    model = HierRNNSummarizer(
        vocab_size=vocab_size, pad_id=pad_id,
        emb_dim=emb_dim, word_hid=word_hid, sent_hid=sent_hid, dec_hid=dec_hid,
        dropout=dropout, use_init_proj=True, two_step=two_step
    ).to(device).eval()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing keys:", missing)
    if unexpected:
        print("[warn] unexpected keys:", unexpected)

    if not text.lstrip().startswith("["):
        text = "[CB] " + text

    enc, enc_token_mask, sent_mask = _build_enc_from_text(text, sp_tok, max_sents, max_words, device)
    with torch.inference_mode():
        gen_ids = model.generate(enc, enc_token_mask, sent_mask, sp_tok, max_len=max_tgt,
                                 top_k=8, no_repeat_ngram_size=3, min_len=10)
    return sp_tok.decode(gen_ids[0].tolist()).strip()


def main():
    p = Path(INPUT_PATH)
    if not p.exists():
        print(f"[error] file not found: {p}")
        return
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        print("[error] input file is empty.")
        return
    print(summarize(text, MODEL_DIR))


if __name__ == "__main__":
    main()
