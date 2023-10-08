#!/usr/bin/env python3
import random
import numpy as np
from transformers import BertConfig, Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import wandb
wandb.login()

data_path = "../dataset/"

dataset_name = "peersum"
model_name = "bert-base-uncased"
max_input_length = 512
batch_size = 4
wandb.init(project="BERT_acceptance_%s_%d"%(dataset_name, max_input_length))
random.seed(42)

def acceptance_categorical(acceptance):
    if "eject" in acceptance:
        return 0
    else:
        return 1

# load dataset
dataset_all = load_dataset('json', data_files=data_path + '%s.json' % dataset_name, split='all')
print("dataset all", len(dataset_all))

# paper_acceptances = set([])
# for sample in dataset_all:
#     paper_acceptances.add(sample["paper_acceptance"])
# print(paper_acceptances)

dataset_train = dataset_all.filter(lambda s: s['label'] == 'train' and s['paper_score']!=-1)
# dataset_train = dataset_train.shuffle(seed=42).select(range(128))
print("dataset train", len(dataset_train))

dataset_val = dataset_all.filter(lambda s: s['label'] == 'val' and s['paper_score']!=-1)
dataset_val = dataset_val.shuffle(seed=42).select(range(512))
print("dataset validation", len(dataset_val))

dataset_test = dataset_all.filter(lambda s: s['label'] == 'test' and s['paper_score']!=-1)
dataset_test = dataset_test.select(random.sample(range(len(dataset_test)), 512))
print("dataset test selected", len(dataset_test))

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
config = BertConfig.from_pretrained(model_name)
# print(config)
config.gradient_checkpointing = False

bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def process_data_to_model_inputs(batch):
    summaries = batch["summary"]
    for i, summary in enumerate(summaries):
        summaries[i] = summary.lower()
    input_dict = tokenizer(
        summaries,
        padding="max_length",
        truncation=True,
        max_length=max_input_length
    )

    results = {}
    results["input_ids"] = input_dict.input_ids
    results["attention_mask"] = input_dict.attention_mask
    labels = batch["paper_acceptance"]
    for i, label in enumerate(labels):
        labels[i] = acceptance_categorical(label)
    results["labels"] = labels
    return results


print("Preprocessing dataset train")
dataset_train = dataset_train.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

print("Preprocessing dataset validation")
dataset_val = dataset_val.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

print("Preprocessing dataset test")
dataset_test = dataset_test.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

# set Python list to PyTorch tensor
dataset_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# set Python list to PyTorch tensor
dataset_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# set Python list to PyTorch tensor
dataset_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

training_args = TrainingArguments(
    optim="adafactor",
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="result",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=1000,
    save_total_limit=3,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    report_to='wandb',
    metric_for_best_model = 'acc',
    greater_is_better=False,
    load_best_model_at_end=True
)


# compute Rouge score during validation
def compute_metrics(pred):
    preds, labels = pred
    return {"acc": accuracy_score(labels, preds.argmax(-1))}


# instantiate trainer
trainer = Trainer(
    model=bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.predict(test_dataset=dataset_test)
# start training
trainer.train()