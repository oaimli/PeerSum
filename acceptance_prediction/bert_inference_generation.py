#!/usr/bin/env python3
import random
import os
import json
import jsonlines
import numpy as np
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score

dataset_name = "peersum_all"
model_name = "result/checkpoint-4000"
max_input_length = 512
batch_size = 4
random.seed(42)
generated_summaries_folder = "/home/miao4/punim0521/NeuralAbstractiveSummarization/reproduced/led_summarization_huggingface/result_peersum_4096_512_paper_hyperparams_new/generated_summaries"

summaries_evaluated_folder = generated_summaries_folder


def acceptance_categorical(acceptance):
    if "eject" in acceptance:
        return 0
    else:
        return 1


for with_disagreements in [[0, 1], [1], [0]]:
    papers = []
    with jsonlines.open("../crawling_data/data/%s.json" % dataset_name, "r") as reader:
        for paper in reader:
            if paper["label"] == "test":
                papers.append(paper)
    print("all original test data", len(papers))
    acceptance_dict = {}
    conflict_samples = {}
    for paper in papers:
        meta_review = paper["meta_review"]
        acceptance_dict[meta_review] = paper["paper_acceptance"]

        review_scores = []
        for i, review_i in enumerate(paper["reviews"]):
            if review_i["rating"] > 0:
                review_scores.append(review_i["rating"])
        range = np.max(review_scores) - np.min(review_scores)

        source_documents = []
        source_documents.append(paper["paper_abstract"])
        for review in paper["reviews"]:
            if "comment" in review.keys():
                text = review["comment"]
                source_documents.append(text)

        if range >= 4:
            conflict_samples[meta_review] = 1
        else:
            conflict_samples[meta_review] = 0

    files = os.listdir(summaries_evaluated_folder)
    print(dataset_name, "dataset test all", len(files))

    dataset_test = []
    for file in files:
        if not os.path.isdir(file):
            with open(os.path.join(summaries_evaluated_folder, file)) as f:
                result = json.load(f)
            acceptance = acceptance_dict[result["reference"]]
            if conflict_samples[result["reference"]] in with_disagreements:
                dataset_test.append({"summary": result["prediction"], "paper_acceptance": acceptance})
    # print(dataset_test)
    print(dataset_name, "dataset test selected", len(dataset_test))
    dataset_test = Dataset.from_list(dataset_test)

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

