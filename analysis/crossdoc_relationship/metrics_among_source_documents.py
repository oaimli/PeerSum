import sys
import random
import multiprocessing
import functools
import numpy as np
import re
import gensim.downloader

sys.path.append("../../../")
from preparing_data.data_loader import loading_mds
from peersum.preparing_data.peersum_loader import loading_peersum
from utils.evaluation import bert_score_similarity, rouge


def gensim_glove(model="glove-wiki-gigaword-300"):
    glove_vectors = gensim.downloader.load(model)
    return glove_vectors


def rouge_variance_multi_process(i, source_documents, rouge):
    c = source_documents[i]
    rouge1s_tmp = []
    rouge2s_tmp = []
    rougels_tmp = []
    for d1_index, d1 in enumerate(c):
        for d2_index, d2 in enumerate(c):
            if d1_index!=d2_index:
                if len(d1.strip().split())>2 and len(d2.strip().split())>2:
                    d1.replace("sentence_split", "\n")
                    d2.replace("sentence_split", "\n")
                    scores = rouge(d1, d2, types=['rouge1', 'rouge2', 'rougeLsum'])
                    rouge1s_tmp.append(scores["rouge1"]["r"])
                    rouge2s_tmp.append(scores["rouge2"]["r"])
                    rougels_tmp.append(scores["rougeLsum"]["f"])
    return {"rouge1_var":np.var(rouge1s_tmp), "rouge2_var":np.var(rouge2s_tmp), "rougel_var":np.var(rougels_tmp), "rouge1_mean":np.mean(rouge1s_tmp), "rouge2_mean":np.mean(rouge2s_tmp), "rougel_mean":np.mean(rougels_tmp)}


def rouge_variance(source_documents):
    count_all = len(source_documents)
    partial_rouge_variance = functools.partial(rouge_variance_multi_process, source_documents=source_documents, rouge=rouge)
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
    print("Rouge 1 Recall variance", np.mean(rouge1s_var), "Rouge 1 Recall mean", np.mean(rouge1s_mean))
    print("Rouge 2 Recall variance", np.mean(rouge2s_var), "Rouge 2 Recall mean", np.mean(rouge2s_mean))
    print("Rouge l F1 variance", np.mean(rougels_var), "Rouge l F1 mean", np.mean(rougels_mean))


def bert_scoring(source_documents):
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    bert_score_variances = []
    bert_score_means = []
    for c in source_documents:
        bert_scores = []
        for d1_index, d1 in enumerate(c):
            for d2_index, d2 in enumerate(c):
                if d2_index>d1_index:
                    bert_scores.append(bert_score_similarity(d1, d2, scorer)["f"])
        bert_score_variances.append(np.var(bert_scores))
        bert_score_means.append(np.mean(bert_scores))
    print("BERTScore variance", np.mean(bert_score_variances), "BERTScore mean", np.mean(bert_score_means))


def wmdistance_variance_multi_process(i, source_documents, word2vec):
    c = source_documents[i]
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


def wmdistance_variance(source_documents, word2vec):
    count_all = len(source_documents)
    partial_wmdistance_variance = functools.partial(wmdistance_variance_multi_process,
                                                      source_documents=source_documents, word2vec=word2vec)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_distances = list(p.imap(partial_wmdistance_variance, range(count_all), chunksize=128))

    distance_variances = []
    distance_means = []
    for item in all_distances:
        distance_variances.append(item["distance_var"])
        distance_means.append(item["distance_mean"])
    print("WMDistance variance", np.mean(distance_variances), "WMDistance mean",
          np.mean(distance_means))


def sampling(source_documents, sampling=1000):
    count_all = len(source_documents)
    indexes = range(count_all)
    if sampling==0:
        return source_documents
    elif sampling>1:
        target_indexes = random.sample(indexes, int(sampling))
        source_documents_tmp = []
        for i in target_indexes:
            source_documents_tmp.append(source_documents[i])
        return source_documents_tmp
    else:
        target_indexes = random.sample(indexes, int(count_all*sampling))
        source_documents_tmp = []
        for i in target_indexes:
            source_documents_tmp.append(source_documents[i])
        return source_documents_tmp


if __name__=="__main__":
    samling_count = 300

    word2vec = gensim_glove("glove-wiki-gigaword-300")

    # print("PeerSum with only official reviews")
    # source_documents, _ = loading_peersum(
    #     "../../../peersum", including_public_comments=False, including_responses=False, data_name="peersum_cleaned", spliting=False)
    # source_documents = sampling(source_documents, samling_count)
    # rouge_variance(source_documents)
    # wmdistance_variance(source_documents, word2vec)
    # bert_scoring(source_documents)

    # print("PeerSum including public comments")
    # source_documents, _ = loading_peersum(
    #     "../../../peersum", including_public_comments=True, including_responses=False, data_name="peersum_cleaned", spliting=False)
    # source_documents = sampling(source_documents, samling_count)
    # rouge_variance(source_documents)
    # wmdistance_variance(source_documents, word2vec)
    # bert_scoring(source_documents)
    #
    # print("PeerSum including public comments and responses")
    # source_documents, _ = loading_peersum(
    #     "../../../peersum", including_public_comments=True, including_responses=True, data_name="peersum_cleaned",
    #     spliting=False)
    # source_documents = sampling(source_documents, samling_count)
    # rouge_variance(source_documents)
    # wmdistance_variance(source_documents, word2vec)
    # bert_scoring(source_documents)

    print("Multi-News")
    # source_documents, _ = loading_mds("../../../preparing_data/", "multinews_cleaned", spliting=False)
    source_documents, _ = loading_mds("/scratch/miao4/datasets_tmp/", "multinews_cleaned", spliting=False)
    source_documents = sampling(source_documents, samling_count)
    rouge_variance(source_documents)
    wmdistance_variance(source_documents, word2vec)
    bert_scoring(source_documents)


    print("Multi-XScience")
    # source_documents, _ = loading_mds("../../../preparing_data/", "multixscience_cleaned", spliting=False)
    source_documents, _ = loading_mds("/scratch/miao4/datasets_tmp/", "multixscience_cleaned", spliting=False)
    source_documents = sampling(source_documents, samling_count)
    rouge_variance(source_documents)
    wmdistance_variance(source_documents, word2vec)
    bert_scoring(source_documents)


    print("Wikisum")
    # source_documents, _ = loading_mds("../../../preparing_data/", "wikisum_cleaned", spliting=False)
    source_documents, _ = loading_mds("/scratch/miao4/datasets_tmp/", "wikisum_cleaned", spliting=False)
    source_documents = sampling(source_documents, samling_count)
    rouge_variance(source_documents)
    wmdistance_variance(source_documents, word2vec)
    bert_scoring(source_documents)


    print("WCEP")
    # source_documents, _ = loading_mds("../../preparing_data/", "wcep_cleaned", spliting=False)
    source_documents, _ = loading_mds("/scratch/miao4/datasets_tmp/", "wcep_cleaned", spliting=False)
    source_documents = sampling(source_documents, samling_count)
    rouge_variance(source_documents)
    wmdistance_variance(source_documents, word2vec)
    bert_scoring(source_documents)