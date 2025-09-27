import os
import math
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_from_disk, DatasetDict
import sentencepiece as spm
from tqdm.auto import tqdm

SEED = 42
torch.backends.cudnn.benchmark = True

SPM_VOCAB_SIZE   = 16000
SPM_MDEL_TYPE    = "unigram"
SPM_CHAR_COV     = 0.9995
SPM_MAX_SENT_LEN = 100000
SP_USER_SYMBOLS  = ["[CB]", "[NEWS]", "[SEP]"]

MAX_SENTS = 50
MAX_WORDS = 32
MAX_TGT   = 128

EMB_DIM   = 384
WORD_HID  = 384
SENT_HID  = 384
DEC_HID   = 768
DROPOUT   = 0.25

TIE_EMBED = True

BATCH_SIZE    = 4
NUM_EPOCHS    = 15
LR            = 2e-3
WEIGHT_DECAY  = 0.01
CLIP_NORM     = 1.0
LABEL_SMOOTH  = 0.1
PATIENCE      = 10
MIN_DELTA     = 5e-4
MIN_EPOCHS    = 8
WARMUP_FRACT  = 0.05
NUM_WORKERS   = 0
AMP           = True
ACCUM_STEPS   = 1

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_re_sent = None
def split_sentences(text: str) -> List[str]:
    global _re_sent
    if _re_sent is None:
        import re
        _re_sent = re.compile(r'(?<=[\.\!\?])\s+')
    text = (text or "").strip()
    parts = _re_sent.split(text)
    return [p.strip() for p in parts if p.strip()]

class SPTok:
    def __init__(self, model_path: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id = 0
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        if self.sp.pad_id() >= 0:
            self.pad_id = self.sp.pad_id()
        else:
            self.pad_id = 0

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i != self.pad_id and i != self.eos_id and i != self.bos_id]
        return self.sp.decode(ids)

class HSumDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict[str, str]], tok: SPTok):
        self.recs = records
        self.tok = tok

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        ex = self.recs[idx]
        text = ex.get("text", "")
        summ = ex.get("summary", "")
        sents = split_sentences(text)[:MAX_SENTS]
        sent_ids = []
        sent_lens = []
        for s in sents:
            ids = self.tok.encode(s)[:MAX_WORDS]
            sent_lens.append(len(ids))
            if len(ids) < MAX_WORDS:
                ids = ids + [self.tok.pad_id]*(MAX_WORDS - len(ids))
            sent_ids.append(ids)
        pad_sent = [self.tok.pad_id]*MAX_WORDS
        if len(sent_ids) < MAX_SENTS:
            sent_ids += [pad_sent]*(MAX_SENTS - len(sent_ids))
            sent_lens += [0]*(MAX_SENTS - len(sent_lens))
        y = self.tok.encode(summ)[:(MAX_TGT-2)]
        y_in  = [self.tok.bos_id] + y
        y_out = y + [self.tok.eos_id]
        if len(y_in) < MAX_TGT:
            y_in  = y_in  + [self.tok.pad_id]*(MAX_TGT - len(y_in))
            y_out = y_out + [self.tok.pad_id]*(MAX_TGT - len(y_out))
        else:
            y_in  = y_in[:MAX_TGT]
            y_out = y_out[:MAX_TGT]
        enc = torch.tensor(sent_ids, dtype=torch.long)
        sent_lens = torch.tensor(sent_lens, dtype=torch.long)
        y_in = torch.tensor(y_in, dtype=torch.long)
        y_out = torch.tensor(y_out, dtype=torch.long)
        return {"enc": enc, "sent_lens": sent_lens, "y_in": y_in, "y_out": y_out}

def collate_fn(batch, pad_id: int):
    enc = torch.stack([b["enc"] for b in batch], dim=0)
    sent_lens = torch.stack([b["sent_lens"] for b in batch], dim=0)
    y_in  = torch.stack([b["y_in"] for b in batch], dim=0)
    y_out = torch.stack([b["y_out"] for b in batch], dim=0)
    enc_token_mask = (enc != pad_id)
    sent_mask = (sent_lens > 0)
    y_mask = (y_out != pad_id)
    return {
        "enc": enc, "enc_token_mask": enc_token_mask, "sent_mask": sent_mask,
        "y_in": y_in, "y_out": y_out, "y_mask": y_mask
    }

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
        w = torch.softmax(e, dim=1)
        s = torch.sum(h * w.unsqueeze(-1), dim=1)
        return s.view(B, S, D)

class HierEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, word_hid, sent_hid, pad_id, dropout=0.25, tie_embed=False):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
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
        self.emb = shared_emb if shared_emb is not None else nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.rnn  = nn.GRU(emb_dim + enc_dim, dec_hid, batch_first=True)
        self.attn = BahdanauAttn(enc_dim, dec_hid)
        self.tie_embed = bool(tie_embed)
        if self.tie_embed:
            self.proj = nn.Linear(dec_hid + enc_dim, emb_dim, bias=False)
            self.out_bias = nn.Parameter(torch.zeros(vocab_size))
            self.out = None
        else:
            self.out  = nn.Linear(dec_hid + enc_dim, vocab_size)

    def forward(self, y_in, H_sent, sent_mask, init_h=None, return_state: bool = False):
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
        if return_state:
            return logits, h
        return logits

class HierRNNSummarizer(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.encoder = HierEncoder(
            vocab_size=vocab_size, emb_dim=EMB_DIM,
            word_hid=WORD_HID, sent_hid=SENT_HID,
            pad_id=pad_id, dropout=DROPOUT
        )
        enc_dim = 2 * SENT_HID
        shared_emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=pad_id)
        self.encoder.emb = shared_emb
        self.decoder = Decoder(
            vocab_size=vocab_size, emb_dim=EMB_DIM, dec_hid=DEC_HID,
            enc_dim=enc_dim, pad_id=pad_id, dropout=DROPOUT, tie_embed=TIE_EMBED,
            shared_emb=shared_emb
        )
        self.init_proj = nn.Linear(enc_dim, DEC_HID)

    def _init_hidden(self, H_sent, sent_mask):
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
    def generate(self, enc, enc_token_mask, sent_mask, sp_tok: SPTok, max_len=MAX_TGT, top_k: int = 8, no_repeat_ngram_size: int = 3, min_len: int = 10):
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

def cosine_with_warmup(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1. + math.cos(math.pi * progress))

def train_one_epoch(model, dl, opt, device, pad_id, scaler=None, step0=0, total_steps=1000, base_lr=LR, warmup_steps=500, label_smooth=0.0, accum_steps=1):
    model.train()
    total, n = 0.0, 0
    step = step0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(dl, total=len(dl), desc="train", leave=False)
    for it, batch in enumerate(pbar, start=1):
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(AMP and scaler is not None)):
            logits = model(batch)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["y_out"].view(-1),
                ignore_index=pad_id,
                label_smoothing=label_smooth,
            ) / max(1, accum_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if it % accum_steps == 0:
            step += 1
            lr = cosine_with_warmup(step, base_lr, warmup_steps, total_steps)
            for g in opt.param_groups: g["lr"] = lr
            if scaler is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step(); opt.zero_grad(set_to_none=True)

        bs = batch["enc"].size(0)
        total += float(loss.item()) * bs * max(1, accum_steps)
        n += bs
        pbar.set_postfix(loss=float(loss.item()) * max(1, accum_steps), lr=f"{opt.param_groups[0]['lr']:.2e}")
    return total / max(1, n), step

@torch.no_grad()
def evaluate_loss(model, dl, device, pad_id, label_smooth=0.1):
    model.eval()
    total, n = 0.0, 0
    pbar = tqdm(dl, total=len(dl), desc="val", leave=False)
    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        logits = model(batch)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["y_out"].view(-1),
            ignore_index=pad_id,
            label_smoothing=label_smooth,
        )
        bs = batch["enc"].size(0)
        total += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=float(loss.item()))
    return total / max(1, n)

def load_dataset_records(data_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    dd = load_from_disk(str(data_dir))
    assert isinstance(dd, DatasetDict)
    def _conv(ds):
        return [{"text": ex["text"], "summary": ex["summary"]} for ex in ds]
    return _conv(dd["train"]), _conv(dd["validation"]), _conv(dd["test"])

def ensure_spm(train_recs: List[Dict], out_dir: Path, force_retrain=False) -> Path:
    spm_path = out_dir / "spm.model"
    if spm_path.exists() and not force_retrain:
        return spm_path
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_corpus = out_dir / "spm_corpus.txt"
    with tmp_corpus.open("w", encoding="utf-8") as f:
        for ex in train_recs:
            f.write(ex["text"].replace("\n", " ") + "\n")
            f.write(ex["summary"].replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=str(tmp_corpus),
        model_prefix=str(out_dir / "spm"),
        vocab_size=SPM_VOCAB_SIZE,
        model_type=SPM_MDEL_TYPE,
        character_coverage=SPM_CHAR_COV,
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        max_sentence_length=SPM_MAX_SENT_LEN,
        user_defined_symbols=",".join(SP_USER_SYMBOLS)
    )
    tmp_corpus.unlink(missing_ok=True)
    return spm_path

def build_dataloaders(train_recs, val_recs, sp_tok, batch_size):
    train_ds = HSumDataset(train_recs, sp_tok)
    val_ds   = HSumDataset(val_recs, sp_tok)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, sp_tok.pad_id)
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, sp_tok.pad_id)
    )
    return train_dl, val_dl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/central_bank_summaries_combined/hf_dataset")
    ap.add_argument("--out_dir", type=str, default="runs/cb_hier_rnn")
    ap.add_argument("--run_name", type=str, default="cb_hier_rnn_v2")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--retrain_sp", action="store_true")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--min_delta", type=float, default=MIN_DELTA)
    ap.add_argument("--min_epochs", type=int, default=MIN_EPOCHS)
    ap.add_argument("--label_smooth", type=float, default=LABEL_SMOOTH)
    ap.add_argument("--accum_steps", type=int, default=ACCUM_STEPS)
    args = ap.parse_args()

    set_seed(SEED)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs, val_recs, test_recs = load_dataset_records(data_dir)
    print(f"Loaded splits: train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

    spm_path = ensure_spm(train_recs, out_dir, force_retrain=args.retrain_sp)
    sp_tok = SPTok(spm_path)
    vocab_size = sp_tok.sp.vocab_size()
    print(f"SentencePiece vocab size: {vocab_size}")

    train_dl, val_dl = build_dataloaders(train_recs, val_recs, sp_tok, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierRNNSummarizer(vocab_size=vocab_size, pad_id=sp_tok.pad_id).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and (device.type == "cuda"))

    total_updates_per_epoch = max(1, len(train_dl) // max(1, args.accum_steps))
    total_steps = total_updates_per_epoch * args.epochs
    warmup_steps = max(500, int(WARMUP_FRACT * total_steps))
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps} | Accum steps: {args.accum_steps}")

    best_val = float("inf")
    no_improve = 0
    step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, step = train_one_epoch(
            model, train_dl, opt, device, sp_tok.pad_id, scaler,
            step0=step, total_steps=total_steps, base_lr=args.lr,
            warmup_steps=warmup_steps, label_smooth=args.label_smooth, accum_steps=args.accum_steps
        )
        dt = time.time() - t0
        val_loss = evaluate_loss(model, val_dl, device, sp_tok.pad_id, label_smooth=args.label_smooth)
        print(f"Epoch {epoch:02d}  {dt:.1f}s  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        improved = (val_loss < best_val - args.min_delta)
        if improved:
            best_val = val_loss
            no_improve = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "pad_id": sp_tok.pad_id,
                "config": {
                    "EMB_DIM": EMB_DIM, "WORD_HID": WORD_HID, "SENT_HID": SENT_HID,
                    "DEC_HID": DEC_HID, "DROPOUT": DROPOUT,
                    "MAX_SENTS": MAX_SENTS, "MAX_WORDS": MAX_WORDS, "MAX_TGT": MAX_TGT
                }
            }
            best_path = out_dir / "best.pt"
            torch.save(ckpt, best_path)
            print(f"  ↳ saved best to {best_path}")
        else:
            no_improve += 1
            if (epoch >= args.min_epochs) and (no_improve >= args.patience):
                print(f"Early stopping (no improvement for {no_improve} evals; min_epochs={args.min_epochs}).")
                break

        if epoch % args.eval_every == 0:
            model.eval()
            ex = val_recs[0]
            one = HSumDataset([ex], sp_tok)[0]
            batch = collate_fn([one], sp_tok.pad_id)
            for k in batch:
                batch[k] = batch[k].to(device)
            gen_ids = model.generate(batch["enc"], batch["enc_token_mask"], batch["sent_mask"], sp_tok, max_len=MAX_TGT)
            gen_text = sp_tok.decode(gen_ids[0].tolist())
            print("\n[Sample]")
            print(">> REF:", ex["summary"][:300])
            print(">> GEN:", gen_text[:300], "\n")

    (out_dir / "tokenizer.model").write_bytes(Path(spm_path).read_bytes())
    torch.save(model.state_dict(), out_dir / "last_weights_only.pt")
    print(f"[DONE] Artifacts in {out_dir}")

if __name__ == "__main__":
    main()
