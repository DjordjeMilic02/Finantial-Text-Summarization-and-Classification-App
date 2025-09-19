import csv, json, random, time, re
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

OUT_DIR  = Path("scratch_han_artifacts"); OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB   = 30000
MIN_FREQ    = 2
LOWERCASE   = True
NORM_NUM    = True
NORM_URL    = True
PAD_IDX, UNK_IDX = 0, 1

MAX_SENTS   = 40
MAX_WORDS   = 30

EMB_DIM     = 128
WORD_HID    = 64
SENT_HID    = 64
DROPOUT     = 0.3

BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 2e-3
WEIGHT_DECAY = 1e-4
CLIP_NORM   = 1.0
PATIENCE    = 5
LABEL_SMOOTH = 0.0

NUM_WORKERS = 0

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

_re_sent = re.compile(r'(?<=[\.\!\?])\s+')
def split_sentences(text: str) -> List[str]:
    parts = _re_sent.split(text.strip())
    parts = [p for p in parts if p.strip()]
    return parts if parts else [text.strip()]

def stratified_split(idxs_by_lbl: Dict[str, List[int]], test_ratio=0.1, val_ratio=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in idxs_by_lbl.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_ratio * n))
        n_val  = int(round(val_ratio * n))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test+n_val].tolist())
        train_idx.extend(idxs[n_test+n_val:].tolist())
    return train_idx, val_idx, test_idx

def stratified_val_split(labels: List[str], val_ratio=0.10, seed=SEED):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i,l in enumerate(labels): by[l].append(i)
    tr_idx, va_idx = [], []
    for l, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n_val = int(round(val_ratio * len(idxs)))
        va_idx.extend(idxs[:n_val].tolist())
        tr_idx.extend(idxs[n_val:].tolist())
    return tr_idx, va_idx

def build_vocab(tokenized_docs: List[List[str]], max_vocab=MAX_VOCAB, min_freq=MIN_FREQ):
    cnt = Counter()
    for toks in tokenized_docs: cnt.update(toks)
    items = [(t,c) for t,c in cnt.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if len(items) > (max_vocab-2): items = items[:max_vocab-2]
    stoi = {tok: i+2 for i,(tok,_) in enumerate(items)}
    stoi["<PAD>"] = PAD_IDX; stoi["<UNK>"] = UNK_IDX
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def encode_tokens(tokens: List[str], stoi: Dict[str,int]) -> List[int]:
    return [stoi.get(t, UNK_IDX) for t in tokens]

def encode_doc(text: str, stoi: Dict[str,int]) -> List[List[int]]:
    sents = split_sentences(text)
    ids_2d = []
    for s in sents[:MAX_SENTS]:
        toks = tokenize(s)
        ids = encode_tokens(toks, stoi)
        if len(ids) < MAX_WORDS:
            ids = ids + [PAD_IDX] * (MAX_WORDS - len(ids))
        else:
            ids = ids[:MAX_WORDS]
        ids_2d.append(ids)
    if len(ids_2d) < MAX_SENTS:
        pad_sent = [PAD_IDX]*MAX_WORDS
        ids_2d += [pad_sent]*(MAX_SENTS - len(ids_2d))
    return ids_2d

class HanDataset(torch.utils.data.Dataset):
    def __init__(self, docs_ids_2d: List[List[List[int]]], labels: List[int], train: bool):
        self.docs = docs_ids_2d
        self.labels = labels
        self.train = train
    def __len__(self): return len(self.docs)
    def __getitem__(self, idx):
        X = self.docs[idx]
        y = self.labels[idx]
        X = torch.tensor(X, dtype=torch.long)
        if self.train:
            keep_prob = 0.9
            mask = (torch.rand(X.size(0)) < keep_prob)
            if mask.any():
                X[~mask] = 0
        return X, torch.tensor(y, dtype=torch.long)

def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys

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
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_IDX)
        self.emb_drop = nn.Dropout(DROPOUT)

        self.word_rnn = nn.GRU(input_size=EMB_DIM, hidden_size=WORD_HID,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.word_attn = WordAttn(in_dim=2*WORD_HID, attn_dim=WORD_HID)

        self.sent_rnn = nn.GRU(input_size=2*WORD_HID, hidden_size=SENT_HID,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.sent_attn = SentAttn(in_dim=2*SENT_HID, attn_dim=SENT_HID)

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(2*SENT_HID, num_classes)

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

def main():
    print(f"Train CSV : {TRAIN_CSV.resolve()}")
    print(f"Test  CSV : {TEST_CSV.resolve()}")
    print(f"Artifacts : {OUT_DIR.resolve()}")

    tr_texts_all, tr_labels_all, tr_sources_all = load_csv(TRAIN_CSV)
    te_texts, te_labels, _ = load_csv(TEST_CSV)

    label2id = {"dovish":0, "neutral":1, "hawkish":2}
    id2label = {v:k for k,v in label2id.items()}
    y_tr_all = [label2id[l] for l in tr_labels_all]
    y_te = [label2id[l] for l in te_labels]
    K = 3

    tr_idx, va_idx = stratified_val_split(tr_labels_all, val_ratio=0.10, seed=SEED)
    def pick(xs, idx): return [xs[i] for i in idx]

    tr_texts = pick(tr_texts_all, tr_idx)
    tr_y      = [y_tr_all[i] for i in tr_idx]
    tr_src    = pick(tr_sources_all, tr_idx)

    va_texts = pick(tr_texts_all, va_idx)
    va_y      = [y_tr_all[i] for i in va_idx]

    print("Sizes:", {"train": len(tr_texts), "val": len(va_texts), "test": len(te_texts)})

    tr_tokens_flat = [tok for t in tr_texts for tok in tokenize(t)]
    stoi, itos = build_vocab([tr_tokens_flat], max_vocab=MAX_VOCAB, min_freq=MIN_FREQ)
    V = max(stoi.values())+1
    print(f"Vocab size: {V} (PAD=0, UNK=1)")

    tr_ids2d = [encode_doc(t, stoi) for t in pbar(tr_texts, desc="Encoding train")]
    va_ids2d = [encode_doc(t, stoi) for t in pbar(va_texts, desc="Encoding val")]
    te_ids2d = [encode_doc(t, stoi) for t in pbar(te_texts, desc="Encoding test")]

    train_ds = HanDataset(tr_ids2d, tr_y, train=True)
    val_ds   = HanDataset(va_ids2d, va_y, train=False)
    test_ds  = HanDataset(te_ids2d, y_te, train=False)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                           collate_fn=collate_batch)
    val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_batch)
    test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_batch)

    counts = np.bincount(np.array(tr_y), minlength=K).astype(np.float32)
    inv = 1.0/np.clip(counts,1.0,None)
    class_weights = torch.tensor(inv/inv.sum()*K, dtype=torch.float32)

    w_src = torch.tensor([0.8 if s=='fomc' else 1.0 for s in tr_src], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HAN(vocab_size=V, num_classes=K).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))

    best_f1, best_path, patience = -1.0, OUT_DIR/"best.pt", PATIENCE

    with (OUT_DIR/"vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "label2id": label2id}, f)

    def run_epoch(loader, train=True):
        if train: model.train()
        else:     model.eval()
        total_loss, seen = 0.0, 0
        all_preds, all_refs = [], []
        it = pbar(loader, total=len(loader), desc="Train" if train else "Eval")
        cw = class_weights.to(device)
        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if train: opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(xb)
                if train:
                    loss_vec = F.cross_entropy(logits, yb, weight=cw, reduction='none')
                    loss = loss_vec.mean()
                else:
                    loss = F.cross_entropy(logits, yb, weight=cw)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update()
            total_loss += float(loss.item()) * xb.size(0); seen += xb.size(0)
            preds = logits.detach().argmax(-1).cpu().numpy()
            refs  = yb.detach().cpu().numpy()
            all_preds.append(preds); all_refs.append(refs)
        preds = np.concatenate(all_preds); refs = np.concatenate(all_refs)
        acc = float((preds==refs).mean())
        f1  = macro_f1(preds, refs, K)
        return total_loss/max(1,seen), acc, f1

    print("Starting training…")
    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(train_dl, train=True)
        va_loss, va_acc, va_f1 = run_epoch(val_dl, train=False)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}  {dt:.1f}s  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}")
        if va_f1 > best_f1:
            best_f1 = va_f1; patience = PATIENCE
            torch.save({"state_dict": model.state_dict(),
                        "vocab_size": V, "num_classes": K,
                        "config": {"EMB_DIM":EMB_DIM,"WORD_HID":WORD_HID,"SENT_HID":SENT_HID,
                                   "DROPOUT":DROPOUT,"MAX_SENTS":MAX_SENTS,"MAX_WORDS":MAX_WORDS}},
                       best_path)
            print(f"  ↳ saved best (val_f1={best_f1:.4f}) to {best_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping."); break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    all_preds, all_refs = [], []
    for xb, yb in pbar(test_dl, total=len(test_dl), desc="Testing"):
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            logits = model(xb)
        preds = logits.argmax(-1).cpu().numpy()
        refs  = yb.numpy()
        all_preds.append(preds); all_refs.append(refs)
    preds = np.concatenate(all_preds); refs = np.concatenate(all_refs)

    test_acc = accuracy(preds, refs); test_f1 = macro_f1(preds, refs, K)
    print(f"\nTEST  acc={test_acc:.4f}  macro-F1={test_f1:.4f}")

    cm = confusion_matrix(preds, refs, K)
    lbls = [id2label[i] for i in range(K)]
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = ["{:>10}".format("")] + ["{:>10}".format(l) for l in lbls]
    print(" ".join(header))
    for i,row in enumerate(cm):
        print(" ".join(["{:>10}".format(lbls[i])] + ["{:>10}".format(v) for v in row]))

    rep = per_class_report(preds, refs, K)
    print("\nPer-class metrics:")
    print("{:>10} {:>10} {:>10} {:>10} {:>10}".format("class","precision","recall","f1","support"))
    for r in rep:
        print("{:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10}".format(lbls[r["class_id"]], r["precision"], r["recall"], r["f1"], r["support"]))

    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"val_best_f1": float(best_f1),
                "test_acc": float(test_acc),
                "test_macro_f1": float(test_f1)}, f, indent=2)

    with (OUT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow([""] + lbls)
        for i, row in enumerate(cm):
            w.writerow([lbls[i]] + list(map(int, row)))

    with (OUT_DIR / "classification_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","precision","recall","f1","support"])
        for r in rep:
            w.writerow([lbls[r["class_id"]],
                        f"{r['precision']:.6f}", f"{r['recall']:.6f}",
                        f"{r['f1']:.6f}", r["support"]])

    print(f"\nSaved reports to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
