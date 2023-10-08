
import gensim
import gensim.downloader
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaModel
import torch
from bert_score import BERTScorer


def bert_score_similarity(text_a, text_b, scorer):
    cands = [text_a]
    refs = [text_b]
    P, R, F = scorer.score(cands, refs)
    return F.mean().item()


# https://huggingface.co/transformers/model_doc/roberta.html#
def roberta_similarity(text_a, text_b, tokenizer, roberta):
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # roberta = RobertaModel.from_pretrained('roberta-base')

    inputs_a = tokenizer(text_a, return_tensors="pt")
    outputs_a = roberta(**inputs_a)
    embedding_a = outputs_a.last_hidden_state

    inputs_b = tokenizer(text_b, return_tensors="pt")
    outputs_b = roberta(**inputs_b)
    embedding_b = outputs_b.last_hidden_state

    # Compute cosine-similarits
    # print(embedding_a[0][0])
    cosine_scores = util.pytorch_cos_sim(embedding_a[0][0], embedding_b[0][0])

    # print(float(cosine_scores))
    return float(cosine_scores)


# https://www.sbert.net/
def sentence_bert_similarity(text_a, text_b, sentence_bert):
    # sentence_bert = SentenceTransformer('all-mpnet-base-v2')

    # Two lists of sentences
    sentences1 = [text_a]
    sentences2 = [text_b]

    #Compute embedding for both lists
    embeddings1 = sentence_bert.encode(sentences1, convert_to_tensor=True)
    embeddings2 = sentence_bert.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # print(float(cosine_scores[0][0]))
    return float(cosine_scores[0][0])


def word_embedding_similarity(text_a, text_b, word2vec):
    # word2vec = gensim.downloader.load("glove-wiki-gigaword-300")
    text_a = [w for w in text_a.split() if word2vec.has_index_for(w)]
    text_b = [w for w in text_b.split() if word2vec.has_index_for(w)]
    if len(text_a)>0 and len(text_b)>0:
        return word2vec.n_similarity(text_a, text_b)
    else:
        return 0


def wmdistance(text_a, text_b, word2vec):
    # word2vec = gensim.downloader.load("glove-wiki-gigaword-300")
    return word2vec.wmdistance(text_a.split(), text_b.split())


def avg_var(similarities):
    print("max", np.max(similarities))
    print("min", np.min(similarities))
    print("avg", np.mean(similarities))
    print("var", np.var(similarities))


def sampling(samples, n):
    count_all =len(samples)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, int(n))
    samples_sampled = []
    for i in target_indexes:
        samples_sampled.append(samples[i])
    return samples_sampled


def analyzing_similarity_snli():
    print("****************SNLI*******************")
    # computing similarities
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    similarity_file = "glue_data/SNLI/test_similarity.json"
    if not os.path.isfile(similarity_file):
        with open("glue_data/SNLI/test.tsv") as f:
            sentence_pairs_all = [line.strip().split("\t") for line in f.readlines()]

        for i, sentence_pair in enumerate(sentence_pairs_all):
            if i == 0:
                sentence_pair.append("bert_score_similarity")
                continue
            text_a = sentence_pair[1]
            text_b = sentence_pair[2]
            sentence_pair.append(bert_score_similarity(text_a, text_b, scorer))

        with open(similarity_file, "w") as f:
            f.write(json.dumps(sentence_pairs_all))
    else:
        with open(similarity_file) as f:
            sentence_pairs_all = json.load(f)
    print("all sentence pairs", len(sentence_pairs_all))

    contras = []
    neutrals = []
    entailments = []
    similarities = []

    for i, sentence_pair in enumerate(sentence_pairs_all):
        if i == 0:
            continue
        similarities.append(sentence_pair[5])
        if sentence_pair[3] == "contradiction":
            contras.append(sentence_pair)
        if sentence_pair[3] == "neutral":
            neutrals.append(sentence_pair)
        if sentence_pair[3] == "entailment":
            entailments.append(sentence_pair)

    # all
    print("all", len(similarities))
    avg_var(similarities)
    samples_sampled = sampling(sentence_pairs_all, 10)
    for s in samples_sampled:
        print(s)

    # contra
    print("contra", len(contras))
    c_similarities = []
    for sentence_pair in contras:
        c_similarities.append(sentence_pair[5])
    avg_var(c_similarities)
    samples_sampled = sampling(sentence_pairs_all, 10)
    for s in samples_sampled:
        print(s)

    # neutral
    print("neutral", len(neutrals))
    c_similarities = []
    for sentence_pair in neutrals:
        c_similarities.append(sentence_pair[5])
    avg_var(c_similarities)
    samples_sampled = sampling(sentence_pairs_all, 10)
    for s in samples_sampled:
        print(s)

    # entailment
    print("entailment", len(entailments))
    c_similarities = []
    for sentence_pair in entailments:
        c_similarities.append(sentence_pair[5])
    avg_var(c_similarities)
    samples_sampled = sampling(sentence_pairs_all, 10)
    for s in samples_sampled:
        print(s)


def analyzing_similarity_mds(data_name):
    print("****************%s*******************"%data_name)
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    test_sampled_file = "glue_data/%s/test_sampled.tsv"%data_name
    if not os.path.isfile(test_sampled_file):
        with open("glue_data/%s/test.tsv"%data_name) as f:
            sentence_pairs_all = [line.strip().split("\t") for line in f.readlines()]
        indexes = range(len(sentence_pairs_all))
        target_indexes = random.sample(indexes[1:], 100000)
        sentence_pairs_sampled = []
        for index in target_indexes:
            sentence_pairs_sampled.append(sentence_pairs_all[index])
        del sentence_pairs_all

        with open("glue_data/%s/sentence_dict.json"%data_name) as f:
            sentence_dict = json.load(f)

        for i, sentence_pair in enumerate(sentence_pairs_sampled):
            text_a = sentence_dict[sentence_pair[1]]
            text_b = sentence_dict[sentence_pair[2]]
            sentence_pair.append(str(bert_score_similarity(text_a, text_b, scorer)))

        sentence_pairs_new = []
        for sentence_pair in sentence_pairs_sampled:
            sentence_pairs_new.append("\t".join(sentence_pair) + "\n")
        with open("glue_data/%s/test_sampled.tsv" % data_name, "w") as f:
            f.writelines(sentence_pairs_new)
    else:
        with open("glue_data/%s/test_sampled.tsv"%data_name) as f:
            sentence_pairs_sampled = [line.strip().split("\t") for line in f.readlines()]
    similarities = []
    for sentence_pair in sentence_pairs_sampled:
        similarities.append(float(sentence_pair[5]))
    avg_var(similarities)

    sentence_pairs_larger = []
    for sentence_pair in sentence_pairs_sampled:
        if float(sentence_pair[5])>0.4:
            sentence_pairs_larger.append(sentence_pair)
    print("Higher similarity", len(sentence_pairs_larger))

    samples_sampled = sampling(sentence_pairs_larger, 10)
    with open("glue_data/%s/sentence_dict.json"%data_name) as f:
        sentence_dict = json.load(f)
    for line in samples_sampled:
        line[1] = sentence_dict[line[1]]
        line[2] = sentence_dict[line[2]]
        line[3] = "none"
    for s in samples_sampled:
        print(s)


if __name__=="__main__":
    analyzing_similarity_snli()
    analyzing_similarity_mds("PEERSUM")
    analyzing_similarity_mds("PEERSUMWO")
    analyzing_similarity_mds("MULTIXSCIENCE")
    analyzing_similarity_mds("MULTINEWS")
    analyzing_similarity_mds("WIKISUM")
    analyzing_similarity_mds("WCEP")