from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

import nltk
import re
import random

dataset = load_dataset("pieetie/pubmed-abstract-summary")["train"].train_test_split(test_size=0.2)
prefixes = ["Summarize: ", "Summarize the following: ", "Give a brief summary: ", "Write a short summary of the following text: ", "Summarize the following text: "]
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
nltk.download("punkt_tab")

def clean_text(text):
    return re.sub(r"<[^<]+>", "", text)

def shuffle_sentences(text):
    sentences = nltk.sent_tokenize(text)
    random.shuffle(sentences)
    return " ".join(sentences)

def tokenize(data):
    abstracts = [clean_text(abstract).strip() for abstract in data["abstract"]] + [shuffle_sentences(clean_text(abstract).strip()) for abstract in data["abstract"]]
    summaries = [clean_text(summary).strip() for summary in data["summary"]] * 2

    model_inputs = tokenizer(abstracts, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(summaries, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(tokenize, batched=True, remove_columns=dataset["test"].column_names)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
model = get_peft_model(model, config)

collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    eval_strategy="epoch",
    label_names=["labels"],
    remove_unused_columns=False,
    learning_rate=1e-3
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator
)

trainer.train()
model = model.merge_and_unload()
model.save_pretrained("flan-t5-small-qa")