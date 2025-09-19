import os, re, csv, json, math, time, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

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

CSV_PATH = Path("combined_dataset.csv")
OUT_DIR  = Path("scratch_textcnn_artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB     = 30000
MIN_FREQ      = 2
LOWERCASE     = True
NORM_NUM      = True
NORM_URL      = True

MAX_LEN_TRAIN = 512
MAX_LEN_EVAL  = 512
EVAL_STRIDE   = 384
PAD_IDX, UNK_IDX = 0, 1

EMB_DIM       = 128
KERNELS       = (2,3,4,5)
CHANNELS      = 128
DROPOUT       = 0.25
HIDDEN_FC     = 256

BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 2e-3
WEIGHT_DECAY  = 1e-4
CLIP_NORM     = 1.0
LABEL_SMOOTH  = 0.05
PATIENCE      = 5

NUM_WORKERS   = 0

def load_csv(path: Path):
    assert path.exists(), f"CSV not found: {path}"
    texts, labels = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        assert {"text","label"}.issubset(rdr.fieldnames or []), "CSV must have columns: text,label"
        for r in rdr:
            t = (r.get("text") or "").strip()
            l = (r.get("label") or "").strip().lower()
            if t and l:
                texts.append(t); labels.append(l)
    return texts, labels

_re_url = re.compile(r'https?://\S+|www\.\S+')
_re_num = re.compile(r'\b\d+(?:\.\d+)?\b')
_re_tok = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
def tokenize(text: str):
    s = text
    if LOWERCASE: s = s.lower()
    if NORM_URL:  s = _re_url.sub(" URL ", s)
    if NORM_NUM:  s = _re_num.sub(" NUM ", s)
    return _re_tok.findall(s)

def stratified_split(texts, labels, test_ratio=0.1, val_ratio=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    by_lbl = defaultdict(list)
    for i, y in enumerate(labels): by_lbl[y].append(i)
    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in by_lbl.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_ratio * n))
        n_val  = int(round(val_ratio * n))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test+n_val].tolist())
        train_idx.extend(idxs[n_test+n_val:].tolist())
    return train_idx, val_idx, test_idx

def build_vocab(tokenized_docs, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ):
    cnt = Counter()
    for toks in tokenized_docs:
        cnt.update(toks)
    items = [(tok, c) for tok, c in cnt.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if len(items) > (max_vocab - 2):
        items = items[:max_vocab - 2]
    stoi = {tok: i+2 for i,(tok,_) in enumerate(items)}
    stoi["<PAD>"] = PAD_IDX
    stoi["<UNK>"] = UNK_IDX
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def encode_tokens(tokens, stoi):
    return [stoi.get(t, UNK_IDX) for t in tokens]

def pad_or_crop(ids, max_len, rng=None):
    if len(ids) <= max_len:
        return ids + [PAD_IDX] * (max_len - len(ids))
    if rng is not None:
        start = rng.randint(0, len(ids) - max_len)
    else:
        start = max(0, (len(ids) - max_len)//2)
    return ids[start:start+max_len]

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, docs_ids, labels, max_len=MAX_LEN_TRAIN, seed=SEED):
        self.docs = docs_ids
        self.labels = labels
        self.max_len = max_len
        self.rng = random.Random(seed)
    def __len__(self): return len(self.docs)
    def __getitem__(self, idx):
        ids = self.docs[idx]
        x = pad_or_crop(ids, self.max_len, rng=self.rng)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class AttnPool1D(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, 1, bias=False)
    def forward(self, H):
        a = torch.tanh(self.w1(H))
        e = self.w2(a).squeeze(-1)
        w = torch.softmax(e, dim=1)
        return torch.sum(H * w.unsqueeze(-1), dim=1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=EMB_DIM,
                 kernels=KERNELS, channels=CHANNELS, dropout=DROPOUT, hidden_fc=HIDDEN_FC):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k//2) for k in kernels])
        self.norms = nn.ModuleList([nn.BatchNorm1d(channels) for _ in kernels])
        self.dropout = nn.Dropout(dropout)
        self.pool = AttnPool1D(dim=channels*len(kernels), hidden=channels)
        self.fc1 = nn.Linear(channels*len(kernels), hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, num_classes)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None: nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        E = self.emb(x)
        z = E.transpose(1, 2)
        Hs, Ts = [], []
        for conv, bn in zip(self.convs, self.norms):
            h = torch.relu(bn(conv(z)))
            Hs.append(h); Ts.append(h.size(-1))
        T_min = min(Ts)
        Hs = [h[..., :T_min] for h in Hs]
        H = torch.cat(Hs, dim=1).transpose(1, 2)
        H = self.dropout(H)
        pooled = self.pool(H)
        u = self.dropout(torch.relu(self.fc1(pooled)))
        logits = self.fc2(u)
        return logits

def smooth_labels(targets, num_classes, eps):
    with torch.no_grad():
        y = torch.zeros((targets.size(0), num_classes), device=targets.device)
        y.fill_(eps / (num_classes - 1))
        y.scatter_(1, targets.view(-1,1), 1 - eps)
        return y

def xent_with_label_smoothing(logits, targets, eps, class_weights=None):
    if eps > 0:
        y = smooth_labels(targets, logits.size(-1), eps)
        logp = torch.log_softmax(logits, dim=-1)
        if class_weights is not None:
            cw = class_weights.view(1, -1)
            loss = -(y * logp * cw).sum(dim=1)
        else:
            loss = -(y * logp).sum(dim=1)
        return loss.mean()
    else:
        return torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)

def accuracy(preds, refs):
    return float((preds == refs).mean())

def macro_f1(preds, refs, K):
    preds = preds.astype(np.int64); refs = refs.astype(np.int64)
    f1s = []
    for c in range(K):
        tp = np.sum((preds == c) & (refs == c))
        fp = np.sum((preds == c) & (refs != c))
        fn = np.sum((preds != c) & (refs == c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2*prec*rec / (prec+rec+1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

def confusion_matrix(preds, refs, K):
    cm = np.zeros((K, K), dtype=int)
    for p, r in zip(preds, refs): cm[r, p] += 1
    return cm

def per_class_report(preds, refs, K):
    rows = []
    for c in range(K):
        tp = np.sum((preds == c) & (refs == c))
        fp = np.sum((preds == c) & (refs != c))
        fn = np.sum((preds != c) & (refs == c))
        tn = np.sum((preds != c) & (refs != c))
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2*prec*rec / (prec+rec+1e-12)
        sup  = tp + fn
        rows.append({"class_id": c, "precision": prec, "recall": rec, "f1": f1, "support": int(sup)})
    return rows

def chunk_ids_eval(ids, max_len=MAX_LEN_EVAL, stride=EVAL_STRIDE):
    if len(ids) <= max_len:
        yield pad_or_crop(ids, max_len, rng=None); return
    start = 0
    while start < len(ids):
        piece = ids[start:start+max_len]
        if len(piece) < max_len:
            piece += [PAD_IDX] * (max_len - len(piece))
        yield piece
        if start + max_len >= len(ids): break
        start += stride

@torch.no_grad()
def predict_doc_logits(model, device, ids):
    chunks = list(chunk_ids_eval(ids))
    x = torch.tensor(chunks, dtype=torch.long, device=device)
    logits = []
    for i in range(0, x.size(0), 64):
        batch = x[i:i+64]
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            l = model(batch)
        logits.append(l.float().cpu().numpy())
    L = np.vstack(logits)
    return L.mean(axis=0)

def eval_docset(model, device, docs_ids, labels, K, desc="Eval"):
    preds, refs = [], []
    iterator = pbar(range(len(docs_ids)), total=len(docs_ids), desc=desc)
    for i in iterator:
        ids, y = docs_ids[i], labels[i]
        logit = predict_doc_logits(model, device, ids)
        preds.append(int(logit.argmax(-1))); refs.append(y)
    preds = np.array(preds); refs = np.array(refs)
    acc = accuracy(preds, refs)
    f1  = macro_f1(preds, refs, K=K)
    return acc, f1, preds, refs

def main():
    print(f"Data CSV   : {CSV_PATH.resolve()}")
    print(f"Artifacts  : {OUT_DIR.resolve()}")

    texts, labels = load_csv(CSV_PATH)
    mapping = {"dovish":0, "neutral":1, "hawkish":2}
    id2label = {0:"dovish", 1:"neutral", 2:"hawkish"}
    y = [mapping[l] for l in labels]
    K = 3

    train_idx, val_idx, test_idx = stratified_split(texts, labels, test_ratio=0.10, val_ratio=0.10, seed=SEED)
    def subset(idxs): return [texts[i] for i in idxs], [y[i] for i in idxs]
    tr_texts, tr_y = subset(train_idx)
    va_texts, va_y = subset(val_idx)
    te_texts, te_y = subset(test_idx)
    print("Sizes:", {"train": len(tr_texts), "val": len(va_texts), "test": len(te_texts)})

    tr_toks = [tokenize(t) for t in tr_texts]
    va_toks = [tokenize(t) for t in va_texts]
    te_toks = [tokenize(t) for t in te_texts]

    stoi, itos = build_vocab(tr_toks, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ)
    V = max(stoi.values()) + 1
    print(f"Vocab size: {V} (PAD=0, UNK=1)")

    tr_ids = [[stoi.get(tok, UNK_IDX) for tok in ts] for ts in tr_toks]
    va_ids = [[stoi.get(tok, UNK_IDX) for tok in ts] for ts in va_toks]
    te_ids = [[stoi.get(tok, UNK_IDX) for tok in ts] for ts in te_toks]

    train_ds = TrainDataset(tr_ids, tr_y, max_len=MAX_LEN_TRAIN, seed=SEED)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )

    counts = np.bincount(np.array(tr_y), minlength=K).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    class_weights = torch.tensor(inv / inv.sum() * K, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TextCNN(vocab_size=V, num_classes=K).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    best_f1, best_path, patience_left = -1.0, OUT_DIR / "best.pt", PATIENCE

    with (OUT_DIR / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "label2id": mapping}, f)

    print("Starting training…")
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        seen = 0
        running_loss = 0.0

        for xb, yb in pbar(train_dl, total=len(train_dl), desc=f"Train epoch {epoch}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(xb)
                loss = xent_with_label_smoothing(logits, yb, LABEL_SMOOTH,
                                                 class_weights=class_weights.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item()) * xb.size(0)
            seen += xb.size(0)

        model.eval()
        val_acc, val_f1, _, _ = eval_docset(model, device, va_ids, va_y, K, desc="Validating")
        dt = time.time() - t0
        train_loss = running_loss / max(1, seen)
        print(f"Epoch {epoch:02d}  time={dt:.1f}s  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = PATIENCE
            torch.save({"state_dict": model.state_dict(),
                        "vocab_size": V, "num_classes": K,
                        "config": {"emb": EMB_DIM, "kernels": KERNELS, "channels": CHANNELS,
                                   "dropout": DROPOUT, "hidden_fc": HIDDEN_FC}},
                       best_path)
            print(f"  ↳ saved new best to {best_path} (val_f1={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    test_acc, test_f1, test_preds, test_refs = eval_docset(model, device, te_ids, te_y, K, desc="Testing")
    print(f"\nTEST  acc={test_acc:.4f}  macro-F1={test_f1:.4f}")

    cm = confusion_matrix(test_preds, test_refs, K)
    labels_order = [id2label[i] for i in range(K)]
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = ["{:>10}".format("")] + ["{:>10}".format(lbl) for lbl in labels_order]
    print(" ".join(header))
    for i, row in enumerate(cm):
        print(" ".join(["{:>10}".format(labels_order[i])] + ["{:>10}".format(v) for v in row]))

    rep = per_class_report(test_preds, test_refs, K)
    print("\nPer-class metrics:")
    print("{:>10} {:>10} {:>10} {:>10} {:>10}".format("class","precision","recall","f1","support"))
    for r in rep:
        print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10}".format(
            id2label[r["class_id"]], r["precision"], r["recall"], r["f1"], r["support"]
        ))

    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"val_best_f1": float(best_f1),
                   "test_acc": float(test_acc),
                   "test_macro_f1": float(test_f1)}, f, indent=2)

    with (OUT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow([""] + labels_order)
        for i, row in enumerate(cm): w.writerow([labels_order[i]] + list(map(int, row)))

    with (OUT_DIR / "classification_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["class","precision","recall","f1","support"])
        for r in rep: w.writerow([id2label[r["class_id"]], f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}", r["support"]])

    print(f"\nSaved reports to {OUT_DIR.resolve()}\n")

if __name__ == "__main__":
    main()
