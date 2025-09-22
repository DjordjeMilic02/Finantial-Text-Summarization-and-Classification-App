import os
from transformers import pipeline

MODEL_PATH = "./finbert-finetuned"

LABEL_MAP = {
    0: "News",
    1: "Earnings Call",
    2: "Central Bank Speech"
}

def main():
    input_file = os.path.join("input", "testInputCompany.txt")
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("[ERROR] No text found in classifierInput.txt")
        return

    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

    result = classifier(text, truncation=True, max_length=512)[0]

    label_id = int(result["label"].replace("LABEL_", ""))
    predicted_class = LABEL_MAP.get(label_id, result["label"])

    print("===================================")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {result['score']:.4f}")
    print("===================================")

if __name__ == "__main__":
    main()
