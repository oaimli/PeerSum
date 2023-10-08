#!/usr/bin/env python3
import random
import os
import json
import jsonlines
import numpy as np
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score

with_disagreements = [0, 1] # 0: without conflict, 1: with conflicts
dataset_name = "peersum"
model_name = "result/checkpoint-4000"
max_input_length = 512
batch_size = 4
random.seed(42)
summaries_evaluated_folder = "ground_truth"


def acceptance_categorical(acceptance):
    if "eject" in acceptance:
        return 0
    else:
        return 1


data_path = "../../datasets/"
dataset_all = load_dataset('json', data_files=data_path + '%s.json' % dataset_name, split='all')
print("dataset all", len(dataset_all))
dataset_test = dataset_all.filter(lambda s: s['label'] == 'test')
dataset_test = dataset_test.select(random.sample(range(len(dataset_test)), 512))
print("dataset test selected", len(dataset_test))

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
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


print("Preprocessing dataset test")
dataset_test = dataset_test.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size
)

# set Python list to PyTorch tensor
dataset_test.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="result",
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
)

test_results = trainer.predict(test_dataset=dataset_test)
print(test_results.metrics)

