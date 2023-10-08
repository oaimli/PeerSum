"""
Use OpenAI embeddings to compute the relevance of summaries
"""

import os
from datasets import load_dataset
import random
import json
import spacy
import openai
from tqdm import tqdm
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# dataset_names = ["peersum", "multinews", "wcep_100", "wikisum", "multixscience"]
dataset_names = ["peersum"]
sampling_num = 32
max_input_tokens = 8192

encoding = tiktoken.get_encoding("cl100k_base")

relevances_sentence_level_all_datasets = []
relevances_summary_level_all_datasets = []
for dataset_name in dataset_names:
    dataset_all = load_dataset('json', data_files= '../../../datasets/%s.json' % dataset_name, split='all')
    dataset_all = list(dataset_all)
    print("all samples", len(dataset_all))

    if dataset_name == "peersum":
        dataset_new = []
        for sample in dataset_all:
            summary = sample["summary"]
            if len(summary.split()) > 80:
                dataset_new.append(sample)
        print("samples after filtering", len(dataset_new))
        dataset_all = dataset_new

    if 0<sampling_num<len(dataset_all):
        # random.seed(42)
        dataset_all = random.sample(dataset_all, sampling_num)

    print("selected samples", dataset_name, len(dataset_all))

    openai.api_key = "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"
    nlp = spacy.load("en_core_web_sm")

    relevances_sentence_level = []
    relevances_summary_level = []
    for i, sample in tqdm(enumerate(dataset_all), total=len(dataset_all)):
        documents = sample["source_documents"]
        # truncate source documents
        documents_truncated = []
        max_tokens_document = int(max_input_tokens/len(documents))
        for document in documents:
            tokens = encoding.encode(document)
            documents_truncated.append(encoding.decode(tokens[:max_tokens_document]))
        documents_sequence = " ".join(documents_truncated)

        source_documents_embedding = []
        while True:
            try:
                # print(len(openai.Embedding.create(input=sentence, model="text-embedding-ada-002")["data"][0]["embedding"]))
                source_documents_embedding = \
                    openai.Embedding.create(input=documents_sequence, model="text-embedding-ada-002")["data"][0][
                        "embedding"]
            except:
                source_documents_embedding = []
            if len(source_documents_embedding) == 1536:
                break

        summary = sample["summary"]
        sentences_summary = []
        summary_words = summary.split()
        for sent in nlp(" ".join(summary_words)).sents:
            sentences_summary.append(sent.text)

        for sentence_summary in sentences_summary:
            sentence_summary_embedding = []
            while True:
                try:
                    sentence_summary_embedding = \
                    openai.Embedding.create(input=sentence_summary, model="text-embedding-ada-002")["data"][0][
                        "embedding"]
                except:
                    sentence_summary_embedding = []
                if len(sentence_summary_embedding) == 1536:
                    break
            relevances_sentence_level.append(cosine_similarity([sentence_summary_embedding], [source_documents_embedding])[0][0])

        summary_embedding = []
        while True:
            try:
                summary_embedding = \
                    openai.Embedding.create(input=summary, model="text-embedding-ada-002")["data"][0][
                        "embedding"]
            except:
                summary_embedding = []
            if len(summary_embedding) == 1536:
                break
        relevances_summary_level.append(
            cosine_similarity([summary_embedding], [source_documents_embedding])[0][0])

    print("sentence level relevance", dataset_name, np.mean(relevances_sentence_level))
    print("summary level relevance", dataset_name, np.mean(relevances_summary_level))
    relevances_sentence_level_all_datasets.append(np.mean(relevances_sentence_level))
    relevances_summary_level_all_datasets.append(np.mean(relevances_summary_level))

print(dataset_names)
print("sentence level relevance:")
print(relevances_sentence_level_all_datasets)
print("summary level relevance:")
print(relevances_summary_level_all_datasets)
print("done")