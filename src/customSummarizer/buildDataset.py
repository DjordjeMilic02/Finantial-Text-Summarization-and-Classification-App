import os
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Value

OUT_DIR = Path("data/central_bank_summaries_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_TOKENS = 200
MIN_SUMMARY_TOKENS = 20
RANDOM_SEED = 42

DATASETS_TO_LOAD = [
    ("tpark-bis/central_bank_speeches", None),
    ("SelmaNajih001/CentralBanksSpeeches-Summary", None),
]

TEXT_CANDIDATES    = ["full_text", "text", "body", "speech", "content", "transcript", "document"]
SUMMARY_CANDIDATES = ["summary", "abstract", "highlights", "short_summary"]

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

def _detect_cols(ds: Dataset) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    cols = list(ds.column_names)
    text_col = _choose_col(cols, TEXT_CANDIDATES)
    sum_col  = _choose_col(cols, SUMMARY_CANDIDATES)

    meta_cols = {}
    for raw_name in cols:
        key = raw_name.lower()
        for cand, norm in META_MAP.items():
            if cand == key or cand in key:
                if norm not in meta_cols:
                    meta_cols[norm] = raw_name
                break
    return text_col, sum_col, meta_cols

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _len_tokens(s: str) -> int:
    return len(_normalize_ws(s).split())

def _unify_split(d: Dataset, source_tag: str) -> Dataset:
    text_col, sum_col, meta_map = _detect_cols(d)
    if not text_col or not sum_col:
        return Dataset.from_list([])

    keep_cols = set(d.column_names)

    def mapper(ex):
        text = _normalize_ws(ex.get(text_col, ""))
        summ = _normalize_ws(ex.get(sum_col, ""))

        meta: Dict[str, Optional[str]] = {
            "title": None, "speaker": None, "affiliation": None,
            "country": None, "date": None
        }

        for norm_key in list(meta.keys()):
            raw = meta_map.get(norm_key)
            if raw and raw in keep_cols:
                val = ex.get(raw)
                if norm_key == "speaker" and isinstance(val, (list, tuple)):
                    val = ", ".join(map(str, val))
                meta[norm_key] = _normalize_ws(str(val)) if val is not None else None

        out = {
            "text": text,
            "summary": summ,
            "title": meta["title"] or "",
            "speaker": meta["speaker"] or "",
            "affiliation": meta["affiliation"] or "",
            "country": meta["country"] or "",
            "date": meta["date"] or "",
            "source": source_tag,
        }
        return out

    d2 = d.map(mapper, remove_columns=d.column_names)

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

def main():
    loaded: List[Dataset] = []

    for name, conf in DATASETS_TO_LOAD:
        try:
            ds_any = load_dataset(name, **(conf or {}))
            base = ds_any["train"] if isinstance(ds_any, DatasetDict) and "train" in ds_any else (next(iter(ds_any.values())) if isinstance(ds_any, DatasetDict) else ds_any)
            print(f"[OK] Loaded {name} with {len(base)} rows.")
            unified = _unify_split(base, source_tag=name)
            print(f"[OK] Mapped {name} -> {len(unified)} rows after filtering.")
            if len(unified) > 0:
                loaded.append(unified)
            else:
                print(f"[WARN] {name}: no usable rows with both text & summary.")
        except Exception as e:
            print(f"[WARN] Could not load '{name}': {type(e).__name__}: {e}")

    if not loaded:
        raise SystemExit("No datasets loaded. Aborting.")

    combined = concatenate_datasets(loaded)
    print(f"[INFO] Combined size before dedupe: {len(combined)}")
    combined = _dedupe(combined)
    print(f"[INFO] Size after dedupe: {len(combined)}")

    combined = combined.shuffle(seed=RANDOM_SEED)
    s1 = combined.train_test_split(test_size=0.10, seed=RANDOM_SEED)
    pool, test_ds = s1["train"], s1["test"]
    s2 = pool.train_test_split(test_size=0.10, seed=RANDOM_SEED)
    train_ds, val_ds = s2["train"], s2["test"]

    dd = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)
    print("Sizes ->", {k: len(v) for k, v in dd.items()})

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
    print("\n[NOTE] Mixed licensing (e.g., CC BY-NC 4.0 on tpark-bis). Follow the strictest license for downstream use.")

if __name__ == "__main__":
    main()
