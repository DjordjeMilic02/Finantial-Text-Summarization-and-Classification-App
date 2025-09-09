import os
import json
from datetime import datetime

BASE_DIR = "customDataset"
OUTPUT_JSONL = "customDataset/parsed_custom_cb.jsonl"

def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return None

    text = " ".join([line.strip() for line in lines[1:] if line.strip()])
    if not text:
        return None

    filename = os.path.basename(filepath)
    date_str = filename.split(".")[0]
    try:
        date = datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%d")
    except Exception:
        date = None

    return {
        "text": text,
        "date": date
    }

def main():
    records = []
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                parsed = parse_file(filepath)
                if parsed:
                    records.append(parsed)

    print(f"[INFO] Parsed {len(records)} central bank statements")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[SUCCESS] Saved parsed dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
