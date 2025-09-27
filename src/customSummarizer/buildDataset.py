import os
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Value, ClassLabel

OUT_DIR = Path("data/central_bank_summaries_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_TOKENS = 200
MIN_SUMMARY_TOKENS = 20
RANDOM_SEED = 42

TEST_FRAC = 0.05
VAL_FRAC  = 0.05

DEFAULT_MULTINEWS_DIR = os.getenv("MULTINEWS_DIR", "multinews").strip()

DATASETS_TO_LOAD = [
    ("tpark-bis/central_bank_speeches", None),
    ("SelmaNajih001/CentralBanksSpeeches-Summary", None),
]

TEXT_CANDIDATES    = ["full_text", "text", "body", "speech", "content", "transcript", "document", "article", "articles"]
SUMMARY_CANDIDATES = ["summary", "abstract", "highlights", "short_summary", "short_abstract"]

META_MAP = {
    "title": "title",
    "speaker": "speaker",
    "speakers": "speaker",
    "affiliation": "affiliation",
    "country_iso2": "country",
    "country": "country",
    "date": "date",
}

def _choose_col(names: List[str], candidates: List[str]) -> Optional[str]:
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    for c in candidates:
        for n in names:
            if c in n.lower():
                return n
    return None

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _len_tokens(s: str) -> int:
    return len(_normalize_ws(s).split())

def _split_sents(txt: str) -> List[str]:
    return [p.strip() for p in re.split(r'(?<=[\.\!\?])\s+', (txt or "").strip()) if p.strip()]

def _postprocess_text_by_source(source_tag: str, text: str) -> str:
    if "multi_news" in source_tag.lower():
        return str(text).replace("|||||", " [SEP] ")
    return text

def _unify_split(d: Dataset, source_tag: str, add_domain_tags: bool, mn_max_summary_sents: int) -> Dataset:
    cols = list(d.column_names)
    text_col = _choose_col(cols, TEXT_CANDIDATES)
    sum_col  = _choose_col(cols, SUMMARY_CANDIDATES)
    if not text_col or not sum_col:
        return Dataset.from_list([])

    keep_cols = set(cols)
    is_mn = "multi_news" in source_tag.lower()
    prefix = "[NEWS] " if is_mn else "[CB] "

    def mapper(ex):
        text_raw = ex.get(text_col, "")
        summ_raw = ex.get(sum_col, "")
        text_raw = _postprocess_text_by_source(source_tag, str(text_raw))

        if is_mn and mn_max_summary_sents > 0:
            sents = _split_sents(str(summ_raw))
            summ_raw = " ".join(sents[:mn_max_summary_sents])

        text = _normalize_ws((prefix + str(text_raw)) if add_domain_tags else str(text_raw))
        summ = _normalize_ws(str(summ_raw))

        meta: Dict[str, Optional[str]] = {
            "title": None, "speaker": None, "affiliation": None,
            "country": None, "date": None
        }
        for norm_key in list(meta.keys()):
            cand = None
            for k in keep_cols:
                lk = k.lower()
                if lk == norm_key or norm_key in lk:
                    cand = k; break
            if cand and cand in keep_cols:
                val = ex.get(cand)
                if norm_key == "speaker" and isinstance(val, (list, tuple)):
                    val = ", ".join(map(str, val))
                meta[norm_key] = _normalize_ws(str(val)) if val is not None else None

        return {
            "text": text,
            "summary": summ,
            "title": meta["title"] or "",
            "speaker": meta["speaker"] or "",
            "affiliation": meta["affiliation"] or "",
            "country": meta["country"] or "",
            "date": meta["date"] or "",
            "source": source_tag,
        }

    d2 = d.map(mapper, remove_columns=cols)

    if "date" in d2.column_names:
        d2 = d2.cast_column("date", Value("string"))

    d2 = d2.filter(lambda ex: _len_tokens(ex["text"]) >= MIN_TEXT_TOKENS and _len_tokens(ex["summary"]) >= MIN_SUMMARY_TOKENS)
    return d2

def _dedupe(ds: Dataset) -> Dataset:
    def _key_fn(example):
        txt = _normalize_ws(example["text"]).lower()
        return {"_key": hash(txt)}
    ds = ds.map(_key_fn)
    seen = set()
    def _uniq(example):
        k = example["_key"]
        if k in seen:
            return False
        seen.add(k)
        return True
    ds = ds.filter(_uniq)
    return ds.remove_columns(["_key"])

def _load_multinews_from_jsonl(base: Path) -> Optional[Dataset]:
    candidates = {
        "train": ["train.jsonl", "train.json"],
        "validation": ["validation.jsonl", "val.jsonl", "validation.json", "val.json"],
        "test": ["test.jsonl", "test.json"],
    }
    data_files: Dict[str, str] = {}
    for split, names in candidates.items():
        for nm in names:
            p = base / nm
            if p.exists():
                data_files[split] = str(p)
                break
    if not data_files:
        return None
    ds_any = load_dataset("json", data_files=data_files)
    parts: List[Dataset] = []
    for split, d in ds_any.items():
        cols = list(d.column_names)
        tcol = _choose_col(cols, ["document", "text", "article", "articles"])
        scol = _choose_col(cols, ["summary", "highlights", "abstract"])
        if not tcol or not scol:
            continue
        std = d.rename_columns({tcol: "text", scol: "summary"})
        parts.append(std)
    if not parts:
        return None
    return concatenate_datasets(parts)

def _pick_first_existing(base: Path, names: List[str]) -> Optional[Path]:
    for nm in names:
        p = base / nm
        if p.exists():
            return p
    head = names[0].split(".")[0]
    kind = names[0].split(".")[1]
    for p in base.glob(f"{head}*.{kind}*"):
        return p
    return None

def _load_multinews_from_src_tgt(base: Path) -> Optional[Dataset]:
    splits = {
        "train": (["train.src", "train.src.cleaned", "train.src.txt"],
                  ["train.tgt", "train.tgt.cleaned", "train.tgt.txt"]),
        "validation": (["val.src", "val.src.cleaned", "validation.src", "validation.src.cleaned"],
                       ["val.tgt", "val.tgt.cleaned", "validation.tgt", "validation.tgt.cleaned"]),
        "test": (["test.src", "test.src.cleaned", "test.src.txt"],
                 ["test.tgt", "test.tgt.cleaned", "test.tgt.txt"]),
    }
    records: List[Dict[str, str]] = []
    found_any = False
    for split, (src_names, tgt_names) in splits.items():
        src_path = _pick_first_existing(base, src_names)
        tgt_path = _pick_first_existing(base, tgt_names)
        if not (src_path and tgt_path and src_path.exists() and tgt_path.exists()):
            continue
        found_any = True
        with src_path.open("r", encoding="utf-8") as fs, tgt_path.open("r", encoding="utf-8") as ft:
            for src_line, tgt_line in zip(fs, ft):
                text = _postprocess_text_by_source("multi_news_local", src_line.rstrip("\n"))
                summ = tgt_line.rstrip("\n")
                records.append({"text": text, "summary": summ})
    if not found_any or not records:
        return None
    return Dataset.from_list(records)

def _maybe_load_local_multinews(path_str: str, add_domain_tags: bool, mn_max_summary_sents: int) -> Optional[Dataset]:
    if not path_str:
        return None
    base = Path(path_str)
    if not base.exists():
        print(f"[WARN] MULTINEWS_DIR not found: {base}")
        return None
    ds = _load_multinews_from_jsonl(base)
    if ds is None:
        ds = _load_multinews_from_src_tgt(base)
    if ds is None:
        print(f"[WARN] Multi-News not found in {base} (expected jsonl or src/tgt files).")
        return None
    ds = _unify_split(ds, source_tag="multi_news_local", add_domain_tags=add_domain_tags, mn_max_summary_sents=mn_max_summary_sents)
    print(f"[OK] Loaded local Multi-News -> {len(ds)} rows after filtering.")
    return ds


def _cast_source_to_string(ds):
    if "source" not in ds.column_names:
        return ds
    feat = ds.features["source"]
    if isinstance(feat, ClassLabel) or (isinstance(feat, Value) and feat.dtype != "string"):
        try:
            return ds.cast_column("source", Value("string"))
        except Exception as e:
            print(f"[WARN] Could not cast 'source' back to string: {type(e).__name__}: {e}")
    return ds

def _cast_source_to_classlabel(ds):
    if "source" not in ds.column_names:
        return ds
    names = sorted(set(ds.unique("source")))
    try:
        return ds.cast_column("source", ClassLabel(names=names))
    except Exception as e:
        print(f"[WARN] Could not cast 'source' to ClassLabel: {type(e).__name__}: {e}")
        return ds

def _safe_stratified_split(ds, test_size: float, seed: int) -> DatasetDict:
    if "source" in ds.column_names:
        ds_lbl = _cast_source_to_classlabel(ds)
        try:
            out = ds_lbl.train_test_split(test_size=test_size, seed=seed, stratify_by_column="source")
            return DatasetDict(
                train=_cast_source_to_string(out["train"]),
                test=_cast_source_to_string(out["test"]),
            )
        except Exception as e:
            print(f"[WARN] Stratified split failed, falling back to random split: {type(e).__name__}: {e}")

    out = ds.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(
        train=_cast_source_to_string(out["train"]),
        test=_cast_source_to_string(out["test"]),
    )

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--multinews_dir", type=str, default=DEFAULT_MULTINEWS_DIR,
                    help="Path to local Multi-News (jsonl or src/tgt). Default 'multinews'.")
    ap.add_argument("--mn_ratio", type=float, default=0.33,
                    help="Target fraction of Multi-News in TRAIN only (0.0–0.9). Final MN count ~ cb_train * mn_ratio / (1 - mn_ratio).")
    ap.add_argument("--mn_max_summary_sents", type=int, default=4,
                    help="Trim Multi-News reference to first N sentences. 0 = no trim.")
    ap.add_argument("--add_domain_tags", action="store_true", default=True,
                    help="Prepend [CB]/[NEWS] tags to text inputs.")
    args = ap.parse_args()

    loaded_cb: List[Dataset] = []
    for name, conf in DATASETS_TO_LOAD:
        try:
            ds_any = load_dataset(name, **(conf or {}))
            base = ds_any["train"] if isinstance(ds_any, DatasetDict) and "train" in ds_any else (next(iter(ds_any.values())) if isinstance(ds_any, DatasetDict) else ds_any)
            print(f"[OK] Loaded {name} with {len(base)} rows.")
            unified = _unify_split(base, source_tag=name, add_domain_tags=args.add_domain_tags, mn_max_summary_sents=args.mn_max_summary_sents)
            print(f"[OK] Mapped {name} -> {len(unified)} rows after filtering.")
            if len(unified) > 0:
                loaded_cb.append(unified)
            else:
                print(f"[WARN] {name}: no usable rows with both text & summary after filtering.")
        except Exception as e:
            print(f"[WARN] Could not load '{name}': {type(e).__name__}: {e}")

    mn_ds = _maybe_load_local_multinews(args.multinews_dir, args.add_domain_tags, args.mn_max_summary_sents) if args.multinews_dir else None

    if not loaded_cb:
        raise SystemExit("No CB datasets loaded. Aborting.")

    cb_all = concatenate_datasets(loaded_cb)
    if mn_ds is not None:
        combined_for_dedupe = concatenate_datasets([cb_all, mn_ds])
        print(f"[INFO] Combined size before dedupe: {len(combined_for_dedupe)}")
        combined_for_dedupe = _dedupe(combined_for_dedupe)
        print(f"[INFO] Size after dedupe: {len(combined_for_dedupe)}")
        cb_all = combined_for_dedupe.filter(lambda ex: "multi_news" not in ex["source"].lower())
        mn_ds  = combined_for_dedupe.filter(lambda ex: "multi_news" in ex["source"].lower())
    else:
        print(f"[INFO] Using CB-only (no Multi-News).")
        cb_all = _dedupe(cb_all)

    cb_all = cb_all.shuffle(seed=RANDOM_SEED)

    s1_cb = _safe_stratified_split(cb_all, test_size=TEST_FRAC, seed=RANDOM_SEED)
    cb_pool, cb_test = s1_cb["train"], s1_cb["test"]

    val_in_pool = VAL_FRAC / (1.0 - TEST_FRAC)
    s2_cb = _safe_stratified_split(cb_pool, test_size=val_in_pool, seed=RANDOM_SEED)
    cb_train, cb_val = s2_cb["train"], s2_cb["test"]

    if mn_ds is not None and args.mn_ratio > 0.0:
        C = len(cb_train)
        target_mn = int(C * args.mn_ratio / max(1e-9, (1.0 - args.mn_ratio)))
        mn_take = min(len(mn_ds), target_mn)
        if mn_take > 0:
            mn_sample = mn_ds.shuffle(seed=RANDOM_SEED).select(range(mn_take))
            train_final = concatenate_datasets([cb_train, mn_sample]).shuffle(seed=RANDOM_SEED)
            print(f"[INFO] Train mix: CB={len(cb_train)}  MN={len(mn_sample)}  total={len(train_final)} (target MN ratio ~ {args.mn_ratio:.2f})")
        else:
            train_final = cb_train
            print(f"[INFO] MN ratio requested but MN size=0 → using CB-only train.")
    else:
        train_final = cb_train
        if mn_ds is None:
            print(f"[INFO] No Multi-News present; using CB-only train.")
        else:
            print(f"[INFO] mn_ratio=0 → CB-only train.")

    dd = DatasetDict(train=train_final, validation=cb_val, test=cb_test)
    sizes = {k: len(v) for k, v in dd.items()}
    print("Sizes ->", sizes)

    hf_out = OUT_DIR / "hf_dataset"
    json_out = OUT_DIR
    hf_out.mkdir(parents=True, exist_ok=True)

    dd.save_to_disk(str(hf_out))
    dd["train"].to_json(str(json_out / "train.jsonl"), orient="records", lines=True, force_ascii=False)
    dd["validation"].to_json(str(json_out / "val.jsonl"), orient="records", lines=True, force_ascii=False)
    dd["test"].to_json(str(json_out / "test.jsonl"), orient="records", lines=True, force_ascii=False)

    print("\n[Peek] One train example:")
    print(json.dumps(dd["train"][0], indent=2, ensure_ascii=False))

    print(f"\n[DONE] Saved to:\n- {hf_out}\n- {json_out / 'train.jsonl'}\n- {json_out / 'val.jsonl'}\n- {json_out / 'test.jsonl'}")
    note = "Mixed licensing (e.g., CC BY-NC 4.0 on BIS; Multi-News is non-commercial). Inherit the strictest license for downstream use."
    print("\n[NOTE]", note)

if __name__ == "__main__":
    main()
