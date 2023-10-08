import sys
import random
import multiprocessing
import functools
import numpy as np
import re
import gensim.downloader

sys.path.append("../../../")
from peersum.loading_data.mds_loader import loading_mds
from peersum.loading_data.peersum_loader import loading_peersum
from utils.metrics import bert_score, rouge, bart_score
from utils.cleaning import cleaning_documents, initializing_spacy


def gensim_glove(model="glove-wiki-gigaword-300"):
    glove_vectors = gensim.downloader.load(model)
    return glove_vectors


def rouge_variance_multi_process(i, source_documents_clusters, rouge, nlp):
    c = cleaning_documents(source_documents_clusters[i], nlp=nlp)

    rouge1s_tmp = []
    rouge2s_tmp = []
    rougels_tmp = []
    for d1_index, d1 in enumerate(c):
        for d2_index, d2 in enumerate(c):
            if d1_index!=d2_index:
                if len(d1.strip().split())>2 and len(d2.strip().split())>2:
                    # d1.replace("sentence_split", "\n")
                    # d2.replace("sentence_split", "\n")
                    scores = rouge(d1, d2, types=['rouge1', 'rouge2', 'rougeLsum'])
                    rouge1s_tmp.append(scores["rouge1"]["fmeasure"])
                    rouge2s_tmp.append(scores["rouge2"]["fmeasure"])
                    rougels_tmp.append(scores["rougeLsum"]["fmeasure"])
    return {"rouge1_var":np.var(rouge1s_tmp), "rouge2_var":np.var(rouge2s_tmp), "rougel_var":np.var(rougels_tmp), "rouge1_mean":np.mean(rouge1s_tmp), "rouge2_mean":np.mean(rouge2s_tmp), "rougel_mean":np.mean(rougels_tmp)}


def rouge_variance(samples):
    source_documents_clusters = []
    for sample in samples:
        source_documents_clusters.append(sample["source_documents"])
    count_all = len(source_documents_clusters)
    partial_rouge_variance = functools.partial(rouge_variance_multi_process, source_documents_clusters=source_documents_clusters, rouge=rouge, nlp=initializing_spacy())
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_rouges = list(p.imap(partial_rouge_variance, range(count_all), chunksize=128))

    rouge1s_var = []
    rouge2s_var = []
    rougels_var = []
    rouge1s_mean = []
    rouge2s_mean = []
    rougels_mean = []
    for item in all_rouges:
        rouge1s_var.append(item["rouge1_var"])
        rouge2s_var.append(item["rouge2_var"])
        rougels_var.append(item["rougel_var"])
        rouge1s_mean.append(item["rouge1_mean"])
        rouge2s_mean.append(item["rouge2_mean"])
        rougels_mean.append(item["rougel_mean"])
    return {"Rouge1-var": np.mean(rouge1s_var), "Rouge1-mean": np.mean(rouge1s_mean), "Rouge2-var": np.mean(rouge2s_var), "Rouge2-mean": np.mean(rouge2s_mean), "RougeLsum-var": np.mean(rougels_var), "RougeLsum-mean": np.mean(rougels_mean)}


def bert_scoring(samples):
    source_documents_clusters = []
    for sample in samples:
        source_documents_clusters.append(sample["source_documents"])
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    bert_score_variances = []
    bert_score_means = []
    for c in source_documents_clusters:
        bert_scores = []
        for d1_index, d1 in enumerate(c):
            for d2_index, d2 in enumerate(c):
                if d2_index>d1_index:
                    bert_scores.append(bert_score(d1, d2, scorer)["f"])
        bert_score_variances.append(np.var(bert_scores))
        bert_score_means.append(np.mean(bert_scores))
    return {"BERTScore variance": np.mean(bert_score_variances), "BERTScore mean": np.mean(bert_score_means)}


def wmdistance_variance_multi_process(i, source_documents_clusters, word2vec):
    c = source_documents_clusters[i]
    distance_cluster = []
    for d1_index, d1 in enumerate(c):
        d1 = re.sub('[,.!?;():\s]', ' ', d1)
        d1 = d1.split()
        for d2_index, d2 in enumerate(c):
            if d2_index > d1_index:
                d2 = re.sub('[,.!?;():\s]', ' ', d2)
                d2 = d2.split()
                if len(d1)>1 and len(d2)>1:
                    distance = word2vec.wmdistance(d1, d2)
                    distance_cluster.append(distance)
    return {"distance_var": np.var(distance_cluster), "distance_mean":np.mean(distance_cluster)}


def wmdistance_variance(samples, word2vec):
    source_documents_clusters = []
    for sample in samples:
        source_documents_clusters.append(sample["source_documents"])
    count_all = len(source_documents_clusters)
    partial_wmdistance_variance = functools.partial(wmdistance_variance_multi_process,
                                                      source_documents_clusters=source_documents_clusters, word2vec=word2vec)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_distances = list(p.imap(partial_wmdistance_variance, range(count_all), chunksize=128))

    distance_variances = []
    distance_means = []
    for item in all_distances:
        distance_variances.append(item["distance_var"])
        distance_means.append(item["distance_mean"])
    return {"WMDistance-var": np.mean(distance_variances), "WMDistance-mean":
          np.mean(distance_means)}


def relevance_score(samples, scorer):
    avgs = []
    variances = []
    for sample in samples:
        source_documents = sample["source_documents"]
        candidates = []
        references = []
        for i, source_document_i in enumerate(source_documents):
            for j, source_document_j in enumerate(source_documents):
                if j!=i:
                    candidates.append(source_document_i)
                    references.append(source_document_j)
        bart_scores = bart_score(candidates, references, bart_scorer=scorer)
        avgs.append(np.mean(bart_scores))
        variances.append(np.var(bart_scores))

    return {"mean": np.mean(avgs), "variance": np.mean(variances)}


def sampling(samples, sampling=1024):
    count_all = len(samples)
    if sampling==0:
        return samples
    elif sampling>1:
        return random.sample(samples, sampling)
    else:
        return random.sample(samples, int(count_all*sampling))

if __name__=="__main__":
    samling_count = 1024

    # word2vec = gensim_glove("glove-wiki-gigaword-300")
    from utils.BARTScore.bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')


    print("PeerSum")
    samples = loading_peersum(
        "../../../peersum", including_author=True, including_abstract=True, including_public=True)
    samples = sampling(samples, samling_count)
    print(rouge_variance(samples))
    print(relevance_score(samples=samples, scorer=bart_scorer))


    # print("Multi-News")
    # samples = loading_mds("../../../datasets/", "multinews")
    # samples = sampling(samples, samling_count)
    # print(rouge_variance(samples))
    # print(relevance_score(samples=samples, scorer=bart_scorer))
    #
    #
    # print("Multi-XScience")
    # samples = loading_mds("../../../datasets/", "multixscience")
    # samples = sampling(samples, samling_count)
    # print(rouge_variance(samples))
    # print(relevance_score(samples=samples, scorer=bart_scorer))
    #
    #
    # print("Wikisum")
    # samples = loading_mds("../../../datasets/", "wikisum")
    # samples = sampling(samples, samling_count)
    # print(rouge_variance(samples))
    # print(relevance_score(samples=samples, scorer=bart_scorer))
    #
    # print("WCEP")
    # samples = loading_mds("../../../datasets/", "wcep_100")
    # samples = sampling(samples, int(samling_count / 2))
    # print(rouge_variance(samples))
    # print(relevance_score(samples=samples, scorer=bart_scorer))