from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def summarize_text():
    model_id = "google/pegasus-large"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,  
        local_files_only=False,  
        dtype="auto",
        use_safetensors=True
    )

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer
    )

    with open("input/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    summary = summarizer(
        text,
        max_length=150,
        min_length=30,
        do_sample=False
    )[0]['summary_text']

    with open("output/summaryPEGASUS.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("Pegasus summary written to output/summaryPEGASUS.txt")

if __name__ == "__main__":
    summarize_text()
