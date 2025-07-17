from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

import re

dataset = load_dataset("pieetie/pubmed-abstract-summary")["train"].train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def clean_text(text):
    return re.sub(r"<[^<]+>", "", text)

def tokenize_closure(tokenizer):
    def tokenize(data):
        abstracts = [clean_text(abstract).strip() for abstract in data["abstract"]]
        summaries = [clean_text(summary) for summary in data["summary"]]

        model_inputs = tokenizer(abstracts, padding="max_length", truncation=True, max_length=512)
        labels = tokenizer(summaries, padding="max_length", truncation=True, max_length=128)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return tokenize

train_dataset = dataset["train"].map(tokenize_closure(tokenizer), batched=True, remove_columns=dataset["train"].column_names)
test_dataset = dataset["test"].map(tokenize_closure(tokenizer), batched=True, remove_columns=dataset["test"].column_names)

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
model.merge_and_unload()
model.save_pretrained("flan-t5-small-qa")