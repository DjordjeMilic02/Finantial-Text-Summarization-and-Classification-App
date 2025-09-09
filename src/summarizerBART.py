from transformers import pipeline

def summarize_text(input_path="input/input.txt", output_path="output/outputBART.txt"):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    summary = summarizer(
        text,
        max_length=200,
        min_length=60,
        do_sample=False
    )[0]['summary_text']

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Summary written to {output_path}")

if __name__ == "__main__":
    summarize_text()
