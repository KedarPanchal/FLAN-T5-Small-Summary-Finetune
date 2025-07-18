from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re

model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-small-qa")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

pipe = pipeline(
    task="summarization",
    model=model,
    tokenizer=tokenizer
)
prompt = input("Abstract: ")
prompt = re.sub(r"<[^<]+>", "", prompt)
print(pipe(prompt, min_new_tokens=10, do_sample=False))