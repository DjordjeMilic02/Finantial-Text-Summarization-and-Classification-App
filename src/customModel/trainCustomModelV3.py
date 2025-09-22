import csv, json, random, time, re, math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
    def pbar(it, **kw): return tqdm(it, **kw)
except Exception:
    def pbar(it, **kw): return it

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TRAIN_CSV = Path("combined_train.csv")
TEST_CSV  = Path("combined_test.csv")
OUT_DIR   = Path("scratch_longtx_artifacts"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB   = 40000
MIN_FREQ    = 2
LOWERCASE   = True
NORM_NUM    = True
NORM_URL    = True
PAD_IDX, UNK_IDX = 0, 1

MAX_LEN_TRAIN  = 2048
MAX_LEN_EVAL   = 2048
EVAL_STRIDE    = 1024

D_MODEL   = 512
N_HEADS   = 8
DEPTH     = 6
MLP_HID   = 2048
DROPOUT   = 0.1

BATCH_SIZE    = 2
GRAD_ACCUM    = 8
EPOCHS        = 15
LR            = 2e-4
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 1e-4
CLIP_NORM     = 1.0
PATIENCE      = 4

NUM_WORKERS   = 0

USE_FOCAL     = False
FOCAL_GAMMA   = 1.5
LABEL_SMOOTH  = 0.05

def load_csv(path: Path):
    assert path.exists(), f"CSV not found: {path}"
    texts, labels, sources = [], [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        assert {"text","label"}.issubset(rdr.fieldnames or []), "CSV must have columns: text,label"
        for r in rdr:
            t = (r.get("text") or "").strip()
            l = (r.get("label") or "").strip().lower()
            s = (r.get("source") or "").strip().lower()
            if t and l:
                texts.append(t); labels.append(l); sources.append(s)
    return texts, labels, sources

_re_url = re.compile(r'https?://\S+|www\.\S+')
_re_num = re.compile(r'\b\d+(?:\.\d+)?\b')
_re_tok = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def tokenize(text: str) -> List[str]:
    s = text
    if LOWERCASE: s = s.lower()
    if NORM_URL:  s = _re_url.sub(" URL ", s)
    if NORM_NUM:  s = _re_num.sub(" NUM ", s)
    return _re_tok.findall(s)

def build_vocab(train_texts: List[str], max_vocab=MAX_VOCAB, min_freq=MIN_FREQ):
    cnt = Counter()
    for t in train_texts: cnt.update(tokenize(t))
    items = [(t,c) for t,c in cnt.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if len(items) > (max_vocab-2): items = items[:max_vocab-2]
    stoi = {tok: i+2 for i,(tok,_) in enumerate(items)}
    stoi["<PAD>"] = PAD_IDX; stoi["<UNK>"] = UNK_IDX
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def encode_ids(text: str, stoi: Dict[str,int]) -> List[int]:
    return [stoi.get(t, UNK_IDX) for t in tokenize(text)]

def pad_or_crop(ids: List[int], max_len: int, rng: np.random.Generator=None) -> List[int]:
    n = len(ids)
    if n <= max_len:
        return ids + [PAD_IDX]*(max_len - n)
    if rng is None:
        start = max(0, (n - max_len)//2)
    else:
        start = int(rng.integers(0, n - max_len + 1))
    return ids[start:start+max_len]

class TrainSeqDataset(torch.utils.data.Dataset):
    def __init__(self, docs_ids: List[List[int]], labels: List[int],
                 max_len=MAX_LEN_TRAIN, seed=SEED):
        assert len(docs_ids) == len(labels)
        self.docs = docs_ids
        self.labels = labels
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.docs)
    def __getitem__(self, i):
        ids = self.docs[i]
        x = pad_or_crop(ids, self.max_len, rng=self.rng)
        y = self.labels[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class ChunkedEvalDataset(torch.utils.data.Dataset):
    def __init__(self, docs_ids: List[List[int]], max_len=MAX_LEN_EVAL, stride=EVAL_STRIDE):
        self.input_ids = []
        self.map = []
        for sid, ids in enumerate(docs_ids):
            n = len(ids)
            if n <= max_len:
                x = ids + [PAD_IDX]*(max_len - n)
                self.input_ids.append(x); self.map.append(sid)
            else:
                start = 0
                while start < n:
                    chunk = ids[start:start+max_len]
                    if len(chunk) < max_len:
                        chunk = chunk + [PAD_IDX]*(max_len - len(chunk))
                    self.input_ids.append(chunk); self.map.append(sid)
                    if start + max_len >= n: break
                    start += stride
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "sample_id": int(self.map[i]),
        }

def collate_ids(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

def collate_chunked(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.long)
    }

def sinusoidal_positions(T: int, D: int, device):
    pe = torch.zeros(T, D, device=device)
    pos = torch.arange(0, T, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32, device=device) * (-math.log(10000.0)/D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

class MultiheadLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def feature_map(self, x):
        return F.elu(x / math.sqrt(self.d_head), alpha=1.0) + 1.0

    def forward(self, x, key_padding_mask=None):
        B,T,D = x.size()
        H, Dh = self.n_heads, self.d_head

        x32 = x.float()
        q = self.q_proj(x32).view(B, T, H, Dh).transpose(1,2)
        k = self.k_proj(x32).view(B, T, H, Dh).transpose(1,2)
        v = self.v_proj(x32).view(B, T, H, Dh).transpose(1,2)

        q = self.feature_map(q)
        k = self.feature_map(k)

        if key_padding_mask is None:
            mask = torch.ones(B, T, device=x.device, dtype=torch.float32)
        else:
            mask = key_padding_mask.float()

        m_q = mask[:, None, :, None]
        m_kv = mask[:, None, :, None]

        q = q * m_q
        k = k * m_kv
        v = v * m_kv

        KV   = torch.einsum("bhtd,bhtf->bhdf", k, v)
        Ksum = k.sum(dim=2)

        num = torch.einsum("bhtd,bhdf->bhtf", q, KV)
        den = torch.einsum("bhtd,bhd->bht",   q, Ksum).unsqueeze(-1)

        eps = 1e-6
        out = num / (den + eps)

        out = out.transpose(1,2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out.to(x.dtype)

class TransformerBlockLA(nn.Module):
    def __init__(self, d_model, n_heads, mlp_hidden, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiheadLinearAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

class LongTXClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_IDX)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD_IDX].zero_()
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([TransformerBlockLA(D_MODEL, N_HEADS, MLP_HID, dropout=DROPOUT) for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.fc   = nn.Linear(D_MODEL, num_classes)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.cached_pe = None

    def add_pos(self, x):
        B,T,D = x.size()
        if (self.cached_pe is None) or (self.cached_pe.size(0) < T) or (self.cached_pe.size(1) != D):
            self.cached_pe = sinusoidal_positions(T, D, device=x.device)
        return x + self.cached_pe[:T]

    def forward(self, x_ids):
        key_padding_mask = (x_ids != PAD_IDX)
        x = self.emb(x_ids)
        x = self.add_pos(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = key_padding_mask.float().unsqueeze(-1)
        s = (x * mask).sum(dim=1)
        z = mask.sum(dim=1).clamp(min=1.0)
        doc = s / z
        logits = self.fc(doc)
        return logits

def accuracy(preds, refs):
    return float((preds == refs).mean())

def macro_f1(preds, refs, K):
    f1s = []
    for c in range(K):
        tp = np.sum((preds==c)&(refs==c))
        fp = np.sum((preds==c)&(refs!=c))
        fn = np.sum((preds!=c)&(refs==c))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def confusion_matrix(preds, refs, K):
    cm = np.zeros((K, K), dtype=int)
    for p, r in zip(preds, refs): cm[r, p] += 1
    return cm

def per_class_report(preds, refs, K):
    rows = []
    for c in range(K):
        tp = np.sum((preds==c)&(refs==c))
        fp = np.sum((preds==c)&(refs!=c))
        fn = np.sum((preds!=c)&(refs==c))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12); sup = tp+fn
        rows.append({"class_id": c, "precision":prec, "recall":rec, "f1":f1, "support":int(sup)})
    return rows

def ce_or_focal(logits, y, weight=None):
    if not USE_FOCAL:
        return F.cross_entropy(logits, y, weight=weight, label_smoothing=LABEL_SMOOTH)
    logp = F.log_softmax(logits, dim=-1)
    ce   = F.nll_loss(logp, y, weight=weight, reduction="none")
    with torch.no_grad():
        p = torch.exp(logp.gather(1, y.unsqueeze(1))).squeeze(1)
    loss = ((1.0 - p) ** FOCAL_GAMMA) * ce
    return loss.mean()

def cosine_with_warmup(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def predict_docs_chunked(model, docs_ids, batch_size=2, window=MAX_LEN_EVAL, stride=EVAL_STRIDE, device=None):
    ds = ChunkedEvalDataset(docs_ids, max_len=window, stride=stride)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_chunked)
    model.eval()
    all_logits = defaultdict(list)
    for batch in pbar(dl, desc="Eval", total=len(dl)):
        sids = batch["sample_id"].tolist()
        x = batch["input_ids"].to(device)
        logits = model(x).detach().cpu().numpy()
        for j, sid in enumerate(sids): all_logits[sid].append(logits[j])
    n_docs = len(docs_ids)
    out = np.zeros((n_docs, model.fc.out_features), dtype=np.float32)
    for sid, chunks in all_logits.items():
        out[sid] = np.stack(chunks, axis=0).mean(axis=0)
    return out

def main():
    print(f"Train CSV : {TRAIN_CSV.resolve()}")
    print(f"Test  CSV : {TEST_CSV.resolve()}")
    print(f"Artifacts : {OUT_DIR.resolve()}")

    tr_texts_all, tr_labels_all, tr_sources_all = load_csv(TRAIN_CSV)
    te_texts, te_labels, te_sources = load_csv(TEST_CSV)

    label2id = {"dovish":0, "neutral":1, "hawkish":2}
    id2label = {v:k for k,v in label2id.items()}
    y_tr_all = [label2id[l] for l in tr_labels_all]
    y_te     = [label2id[l] for l in te_labels]
    K = 3

    def stratified_val_split(labels, val_ratio=0.10, seed=SEED):
        rng = np.random.default_rng(seed)
        by = defaultdict(list)
        for i,l in enumerate(labels): by[l].append(i)
        tr_idx, va_idx = [], []
        for l, idxs in by.items():
            idxs = np.array(idxs); rng.shuffle(idxs)
            n_val = int(round(val_ratio * len(idxs)))
            va_idx.extend(idxs[:n_val].tolist()); tr_idx.extend(idxs[n_val:].tolist())
        return tr_idx, va_idx

    tr_idx, va_idx = stratified_val_split(tr_labels_all, val_ratio=0.10, seed=SEED)
    def pick(xs, idx): return [xs[i] for i in idx]
    tr_texts = pick(tr_texts_all, tr_idx); tr_y = [y_tr_all[i] for i in tr_idx]
    va_texts = pick(tr_texts_all, va_idx); va_y = [y_tr_all[i] for i in va_idx]

    print("Sizes:", {"train": len(tr_texts), "val": len(va_texts), "test": len(te_texts)})

    stoi, itos = build_vocab(tr_texts, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ)
    V = max(stoi.values()) + 1
    print(f"Vocab size: {V} (PAD=0, UNK=1)")

    tr_ids = [encode_ids(t, stoi) for t in pbar(tr_texts, desc="Encode train")]
    va_ids = [encode_ids(t, stoi) for t in pbar(va_texts, desc="Encode val")]
    te_ids = [encode_ids(t, stoi) for t in pbar(te_texts, desc="Encode test")]

    train_ds = TrainSeqDataset(tr_ids, tr_y, max_len=MAX_LEN_TRAIN, seed=SEED)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                           collate_fn=collate_ids)

    counts = np.bincount(np.array(tr_y), minlength=K).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    class_weights = torch.tensor(inv / inv.sum() * K, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LongTXClassifier(vocab_size=V, num_classes=K).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * (len(train_dl) // max(1, GRAD_ACCUM))
    warmup_steps = int(WARMUP_RATIO * total_steps) if total_steps > 0 else 0

    best_f1, best_path, patience = -1.0, OUT_DIR/"best.pt", PATIENCE
    with (OUT_DIR/"vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "label2id": label2id}, f)

    print("Starting training…")
    step = 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        it = pbar(train_dl, total=len(train_dl), desc=f"Train ep{epoch}")
        for i, (xb, yb) in enumerate(it, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = ce_or_focal(logits, yb, weight=class_weights.to(device))
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: loss is NaN/Inf; skipping batch.")
                opt.zero_grad(set_to_none=True)
                continue
            loss = loss / GRAD_ACCUM
            loss.backward()
            if i % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                step += 1
                lr = cosine_with_warmup(step, total_steps, warmup_steps, LR)
                for g in opt.param_groups: g['lr'] = lr
                opt.step(); opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * xb.size(0) * GRAD_ACCUM
            seen += xb.size(0)
            it.set_postfix(loss=f"{running/max(1,seen):.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        model.eval()
        va_logits = predict_docs_chunked(model, va_ids, batch_size=max(1,BATCH_SIZE), device=device)
        va_preds = va_logits.argmax(axis=-1)
        va_acc = accuracy(va_preds, np.array(va_y)); va_f1 = macro_f1(va_preds, np.array(va_y), K)
        pred_dist = np.bincount(va_preds, minlength=K).tolist()
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}  {dt:.1f}s  train_loss={running/max(1,seen):.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")
        print(f"  Val pred dist: {pred_dist}")

        if va_f1 > best_f1:
            best_f1 = va_f1; patience = PATIENCE
            torch.save({"state_dict": model.state_dict(),
                        "vocab_size": V, "num_classes": K,
                        "config": {"D_MODEL":D_MODEL,"N_HEADS":N_HEADS,"DEPTH":DEPTH,
                                   "MLP_HID":MLP_HID,"DROPOUT":DROPOUT,
                                   "MAX_LEN_TRAIN":MAX_LEN_TRAIN,"MAX_LEN_EVAL":MAX_LEN_EVAL}},
                       best_path)
            print(f"  ↳ saved best (val_f1={best_f1:.4f}) to {best_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    te_logits = predict_docs_chunked(model, te_ids, batch_size=max(1,BATCH_SIZE), device=device)
    te_preds = te_logits.argmax(axis=-1)
    te_refs  = np.array(y_te)

    test_acc = accuracy(te_preds, te_refs); test_f1 = macro_f1(te_preds, te_refs, K)
    print(f"\nTEST  acc={test_acc:.4f}  macro-F1={test_f1:.4f}")

    cm = confusion_matrix(te_preds, te_refs, K)
    lbls = [id2label[i] for i in range(K)]
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = ["{:>10}".format("")] + ["{:>10}".format(l) for l in lbls]
    print(" ".join(header))
    for i,row in enumerate(cm):
        print(" ".join(["{:>10}".format(lbls[i])] + ["{:>10}".format(v) for v in row]))

    rep = per_class_report(te_preds, te_refs, K)
    print("\nPer-class metrics:")
    print("{:>10} {:>10} {:>10} {:>10} {:>10}".format("class","precision","recall","f1","support"))
    for r in rep:
        print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10}".format(lbls[r["class_id"]],
                                                                    r["precision"], r["recall"], r["f1"], r["support"]))

    with (OUT_DIR/"metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"val_best_f1": float(best_f1),
                   "test_acc": float(test_acc),
                   "test_macro_f1": float(test_f1)}, f, indent=2)
    with (OUT_DIR/"confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow([""]+lbls)
        for i,row in enumerate(cm): w.writerow([lbls[i]] + list(map(int,row)))
    with (OUT_DIR/"classification_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for r in rep: w.writerow([lbls[r["class_id"]],
                                  f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                                  f"{r['f1']:.6f}", r["support"]])
    print(f"\nSaved reports to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
