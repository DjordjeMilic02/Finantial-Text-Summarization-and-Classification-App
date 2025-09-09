from transformers import pipeline

def summarize_text():
    summarizer = pipeline("summarization", model="t5-base", device=0)

    with open("input/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text = "summarize: " + text

    summary = summarizer(
        text,
        max_length=200,
        min_length=60,
        do_sample=False
    )[0]['summary_text']

    with open("output/summaryT5.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("T5 summary written to output/summary_t5.txt")

if __name__ == "__main__":
    summarize_text()
