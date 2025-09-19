import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from datasets import load_dataset, get_dataset_config_names
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

CUSTOM_ROOT = Path("customDataset")
OUT_ALL     = Path("combined_dataset.csv")
OUT_TRAIN   = Path("combined_train.csv")
OUT_TEST    = Path("combined_test.csv")

INCLUDE_FOMC = True
INCLUDE_ECB  = True

BALANCE_TRAIN_ONLY = True
BALANCE_ALL        = False

SEED = 42
np.random.seed(SEED)

CURRENCIES = {"AUD","CAD","CHF","EUR","GBP","JPY","NZD","USD"}
FLOAT_LINE_RE = re.compile(r"^[\s\+\-]?[0-9]+(?:[.,][0-9]+)?(?:\s*%?)\s*$")

def read_custom_file(fp: Path) -> Tuple[Optional[float], str]:
    """Reads one .txt file: first non-empty line as float (if parseable), rest as text."""
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    first_idx = next((i for i,ln in enumerate(lines) if ln.strip() != ""), None)
    if first_idx is None:
        return (None, "")
    first = lines[first_idx].strip()
    val = parse_float_maybe(first)
    text = "\n".join(lines[first_idx+1:]).strip()
    return (val, text)

def parse_float_maybe(s: str) -> Optional[float]:
    t = s.strip().replace(",", "").replace("%","")
    try:
        return float(t)
    except Exception:
        return None

def parse_date_from_filename(name: str) -> Optional[str]:
    base = name.split(".")[0]
    if len(base) != 8 or not base.isdigit():
        return None
    dd, mm, yyyy = base[:2], base[2:4], base[4:]
    return f"{yyyy}-{mm}-{dd}"

def scan_custom_root(root: Path) -> List[Dict]:
    """Scan customDataset/(CUR)/year/ddmmyyyy.txt -> rows with value, text, currency, date."""
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root.resolve()}")
    rows = []
    for cur_dir in root.iterdir():
        if not cur_dir.is_dir(): continue
        cur = cur_dir.name.upper()
        if cur not in CURRENCIES: continue
        for year_dir in cur_dir.iterdir():
            if not year_dir.is_dir(): continue
            for fp in year_dir.glob("*.txt"):
                val, text = read_custom_file(fp)
                if not text:
                    continue
                date_iso = parse_date_from_filename(fp.name) or ""
                rows.append({
                    "source": "custom",
                    "currency": cur,
                    "date": date_iso,
                    "value": val,
                    "text": text
                })
    return rows

def impute_values(rows: List[Dict]) -> None:
    """Impute missing 'value' with currency mean; fallback to global mean."""
    per_cur_vals: Dict[str, List[float]] = {}
    for r in rows:
        if r["value"] is not None:
            per_cur_vals.setdefault(r["currency"], []).append(float(r["value"]))
    per_cur_mean = {c: (float(np.mean(vs)) if len(vs)>0 else None) for c,vs in per_cur_vals.items()}
    all_vals = [float(r["value"]) for r in rows if r["value"] is not None]
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0
    for r in rows:
        if r["value"] is None:
            r["value"] = per_cur_mean.get(r["currency"]) if per_cur_mean.get(r["currency"]) is not None else global_mean

def compute_tertiles(values: np.ndarray) -> Tuple[float,float]:
    """Global tertiles for ~equal 1/3 splits across all currencies."""
    q1 = float(np.nanquantile(values, 1/3))
    q2 = float(np.nanquantile(values, 2/3))
    return q1, q2

def label_from_value(v: float, low_thr: float, high_thr: float) -> str:
    if v < low_thr:  return "dovish"
    if v > high_thr: return "hawkish"
    return "neutral"

def load_fomc_rows() -> List[Dict]:
    """Use ALL available splits; map label_text; ignore originals when splitting later."""
    if not HAS_DATASETS:
        print("WARNING: 'datasets' not installed; skipping FOMC.")
        return []
    rows = []
    for split in ("train", "test"):
        try:
            ds = load_dataset("FinanceMTEB/FOMC", split=split)
        except Exception:
            continue
        for ex in ds:
            text = (ex.get("sentence") or "").strip()
            lbl  = (ex.get("label_text") or "").strip().lower()
            if not text or not lbl:
                continue
            if lbl in ("dove","accommodative"): lbl = "dovish"
            if lbl in ("hawk","restrictive","tight"): lbl = "hawkish"
            if lbl not in ("dovish","neutral","hawkish"):
                continue
            rows.append({
                "source": "fomc",
                "currency": "",
                "date": "",
                "value": None,
                "text": text,
                "label": lbl
            })
    return rows

def load_ecb_rows() -> List[Dict]:
    """Gather ALL configs/splits from gtfintechlab/european_central_bank, skip 'irrelevant'."""
    if not HAS_DATASETS:
        print("WARNING: 'datasets' not installed; skipping ECB.")
        return []
    try:
        configs = get_dataset_config_names("gtfintechlab/european_central_bank")
    except Exception as e:
        print(f"WARNING: Could not list ECB configs: {e}")
        return []
    rows = []
    for cfg in configs:
        for split in ("train", "validation", "val", "dev", "test"):
            try:
                ds = load_dataset("gtfintechlab/european_central_bank", cfg, split=split)
            except Exception:
                continue
            for ex in ds:
                text = ""
                for key in ("sentences", "sentence", "text", "content", "document"):
                    if key in ex and (ex.get(key) or "").strip():
                        text = (ex.get(key) or "").strip()
                        break
                if not text:
                    continue
                lbl_raw = None
                for key in ("stance_label", "label", "stance"):
                    if key in ex and ex.get(key) is not None:
                        lbl_raw = ex.get(key); break
                if lbl_raw is None:
                    continue
                if isinstance(lbl_raw, (int, np.integer)):
                    continue
                lbl = str(lbl_raw).strip().lower()
                if lbl in ("irrelevant", "", "unknown", "n/a"):
                    continue
                if lbl in ("dove","accommodative"): lbl = "dovish"
                elif lbl in ("hawk","restrictive","tight"): lbl = "hawkish"
                if lbl not in ("dovish","neutral","hawkish"):
                    continue
                rows.append({
                    "source": "ecb",
                    "currency": "EUR",
                    "date": "",
                    "value": None,
                    "text": text,
                    "label": lbl
                })
    return rows

def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["text","label","source","currency","date","value"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

def stratified_split_indices(labels: List[str], test_ratio=0.10, seed=SEED):
    """Return (train_idx, test_idx) stratified by label."""
    rng = np.random.default_rng(seed)
    by_lbl: Dict[str, List[int]] = {}
    for i, y in enumerate(labels):
        by_lbl.setdefault(y, []).append(i)
    train_idx, test_idx = [], []
    for y, idxs in by_lbl.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_ratio * n))
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    return train_idx, test_idx

def count_labels(rows: List[Dict]) -> Dict[str,int]:
    d: Dict[str,int] = {}
    for r in rows:
        y = r.get("label","")
        if y: d[y] = d.get(y,0) + 1
    return d

def oversample_to_balance(rows: List[Dict], seed=SEED) -> List[Dict]:
    if not rows:
        return rows
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        y = r.get("label","")
        if y: by.setdefault(y, []).append(r)
    if not by:
        return rows
    rng = np.random.default_rng(seed)
    max_n = max(len(lst) for lst in by.values())
    out: List[Dict] = []
    for y, lst in by.items():
        k = len(lst)
        if k == 0:
            continue
        if k == max_n:
            out.extend(lst)
        else:
            need = max_n - k
            idxs = rng.integers(0, k, size=need)
            picks = [lst[i].copy() for i in idxs]
            out.extend(lst + picks)
    rng.shuffle(out)
    return out

def main():
    print(f"Scanning custom dataset at: {CUSTOM_ROOT.resolve()}")
    custom_rows = scan_custom_root(CUSTOM_ROOT)
    if not custom_rows:
        raise SystemExit("No custom samples found. Check folder layout and contents.")
    print(f"Custom samples (raw): {len(custom_rows)}")

    impute_values(custom_rows)
    values = np.array([float(r["value"]) for r in custom_rows], dtype=np.float64)
    low_thr, high_thr = compute_tertiles(values)
    print(f"Tertiles from custom moves: low={low_thr:.6f}, high={high_thr:.6f}")
    for r in custom_rows:
        r["label"] = label_from_value(float(r["value"]), low_thr, high_thr)

    print("Custom label counts:", count_labels(custom_rows))

    all_rows = list(custom_rows)

    if INCLUDE_FOMC:
        print("Including FinanceMTEB/FOMC…")
        fomc_rows = load_fomc_rows()
        print(f"FOMC samples gathered: {len(fomc_rows)}")
        all_rows.extend(fomc_rows)

    if INCLUDE_ECB:
        print("Including gtfintechlab/european_central_bank…")
        ecb_rows = load_ecb_rows()
        print(f"ECB samples (stance != irrelevant): {len(ecb_rows)}")
        all_rows.extend(ecb_rows)

    print("Final combined label counts:", count_labels(all_rows))
    print(f"Total combined rows: {len(all_rows)}")

    if BALANCE_ALL:
        print("Balancing FULL dataset by oversampling to equal class counts…")
        before = count_labels(all_rows)
        all_rows = oversample_to_balance(all_rows, seed=SEED)
        after  = count_labels(all_rows)
        print(f"Balanced FULL counts: {after} (was {before})")

    write_csv(all_rows, OUT_ALL)
    print(f"Wrote ALL rows to {OUT_ALL.resolve()}")

    labels_all = [r["label"] for r in all_rows]
    train_idx, test_idx = stratified_split_indices(labels_all, test_ratio=0.10, seed=SEED)
    train_rows = [all_rows[i] for i in train_idx]
    test_rows  = [all_rows[i] for i in test_idx]

    print(f"Train size: {len(train_rows)} | label counts: {count_labels(train_rows)}")
    print(f"Test  size: {len(test_rows)}  | label counts: {count_labels(test_rows)}")

    if BALANCE_TRAIN_ONLY:
        print("Balancing TRAIN split by oversampling to equal class counts…")
        before = count_labels(train_rows)
        train_rows = oversample_to_balance(train_rows, seed=SEED)
        after  = count_labels(train_rows)
        print(f"Balanced TRAIN counts: {after} (was {before})")
    else:
        print("Leaving TRAIN split unbalanced (BALANCE_TRAIN_ONLY=False).")

    write_csv(train_rows, OUT_TRAIN)
    write_csv(test_rows, OUT_TEST)
    print(f"Wrote TRAIN to {OUT_TRAIN.resolve()}")
    print(f"Wrote TEST  to {OUT_TEST.resolve()}")

if __name__ == "__main__":
    main()
