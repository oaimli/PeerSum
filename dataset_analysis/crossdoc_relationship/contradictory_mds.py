"""
This is only for the construction of sentence pairs. Sentence embedding and contradictory prediction are based on "SemBERT-master".
"""
import sys
import re
import gensim.downloader
import random
import multiprocessing
import functools
import json
import os
from tqdm import tqdm

sys.path.append("../../../")
from peersum.loading_data.mds_loader import loading_mds
from peersum.loading_data.peersum_loader import loading_peersum


def get_pairs(document_i, document_j, document_i_index, document_j_index, sentence_pair_index, cluster_index):
    sentence_pairs = []
    for s1_index, s1 in enumerate(document_i.split("sentence_split")):
        s1_split = s1.split()
        for s2_index, s2 in enumerate(document_j.split("sentence_split")):
            s2_split = s2.split()
            if len(s1_split)>2 and len(s2_split)>2:# add more constraints
                sentence_pairs.append([str(sentence_pair_index), str(cluster_index) + "_" + str(document_i_index) + "_" + str(s1_index), str(cluster_index) + "_" + str(document_j_index) + "_" + str(s2_index), "neutral", str(cluster_index)])
    return sentence_pairs


def sentence_pair_generating_multi_process(i, document_clusters):
    sentence_dict = {}
    sentence_pairs = []
    documents = document_clusters[i]
    document_count = len(documents)
    for m in range(document_count):
        document_m = documents[m]
        document_m = re.sub('[():\s]+', ' ', document_m)

        for n in range(m + 1, document_count):
            document_n = documents[n]
            document_n = re.sub('[():\s]+', ' ', document_n)
            sentence_pairs_tmp = get_pairs(document_m, document_n, m, n, -1, i)
            sentence_pairs.extend(sentence_pairs_tmp)

        for s_index, s in enumerate(document_m.split("sentence_split")):
            sentence_dict[str(i) + "_" + str(m) + "_" + str(s_index)] = s

    return sentence_pairs, sentence_dict


def sentence_pair_generating(document_clusters, dataset_name):
    count_all = len(document_clusters)
    partial_sentence_pair_generating = functools.partial(sentence_pair_generating_multi_process,
                                                    document_clusters=document_clusters)
    with multiprocessing.Pool(8) as p:
        results = list(tqdm(p.imap(partial_sentence_pair_generating, range(count_all), chunksize=2), total=count_all, desc="sentence pair generating"))

    pairs = []
    sentence_dict = {}
    for r in results:
        pairs.extend(r[0])
        sentence_dict.update(r[1])

    for i, pair in enumerate(pairs):
        pair[0] = str(i)

    sentence_pairs = []
    sentence_pairs.append(
        "index" + "\t" + "sentence_1" + "\t" + "sentence_2" + "\t" + "gold_label" + "\t" + "cluster_index" + "\n")
    for pair in pairs:
        sentence_pairs.append("\t".join(pair) + "\n")

    print("all sentence pairs", len(sentence_pairs))
    with open("../../baselines/sembert/SemBERT-master/glue_data/%s/test.tsv"%dataset_name, "w") as f:
        f.writelines(sentence_pairs)

    print("all sentences", len(sentence_dict))
    with open("../../baselines/sembert/SemBERT-master/glue_data/%s/sentence_dict.json" % dataset_name, "w") as f:
        f.write(json.dumps(sentence_dict))


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
    # sentence pairs generation, prediction refer to SemBERT
    # print("PeerSum with public comments")
    # source_documents, _ = loading_peersum(
    #     "../../peersum", including_public_comments=True, data_name="peersum_cleaned", spliting=False)
    # source_documents = sampling(source_documents, 2000)
    # sentence_pair_generating(source_documents, "PEERSUM")
    # del source_documents
    #
    # print("PeerSum without public comments")
    # source_documents, _ = loading_peersum(
    #     "../../peersum", including_public_comments=False, data_name="peersum_cleaned", spliting=False)
    # source_documents = sampling(source_documents, 2000)
    # sentence_pair_generating(source_documents, "PEERSUMWO")
    # del source_documents
    #
    # print("Multi-News")
    # source_documents, _ = loading_mds("../../preparing_data/", "multinews_cleaned", spliting=False)
    # source_documents = sampling(source_documents, 2000)
    # sentence_pair_generating(source_documents, "MULTINEWS")
    # del source_documents
    #
    #
    # print("Multi-XScience")
    # source_documents, _ = loading_mds("../../preparing_data/", "multixscience_cleaned", spliting=False)
    # source_documents = sampling(source_documents, 2000)
    # sentence_pair_generating(source_documents, "MULTIXSCIENCE")
    # del source_documents
    #
    #
    # print("Wikisum lemma stop")
    # source_documents, _ = loading_mds(
    #     "../../preparing_data/", "wikisum_cleaned", spliting=False)
    # source_documents = sampling(source_documents, 2000)
    # sentence_pair_generating(source_documents, "WIKISUM")
    # del source_documents


    print("WCEP")
    source_documents, _ = loading_mds("../../preparing_data/", "wcep_cleaned", spliting=False)

    sampled_clusters_file = "../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/sampled_clusters.json"
    if os.path.isfile(sampled_clusters_file):
        with open(sampled_clusters_file) as f:
            source_documents = json.load(f)
    else:
        source_documents = sampling(source_documents, 2000)
        with open(sampled_clusters_file, "w") as f:
            f.write(json.dumps(source_documents))
    print("sampled document clusters", len(source_documents))

    sentence_pairs = []
    sentence_dict = {}
    sentence_pairs.append(
        "index" + "\t" + "sentence_1" + "\t" + "sentence_2" + "\t" + "gold_label" + "\t" + "cluster_index" + "\n")
    with open("../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/test.tsv",
              "w") as f:
        f.writelines(sentence_pairs)
    sentence_pair_index = 0
    for c_index, c in enumerate(source_documents):
        print(c_index)
        sentence_pairs_cluster = []
        for d1_index, d1 in enumerate(c):
            d1 = re.sub('[():\s]+', ' ', d1)
            for d2_index, d2 in enumerate(c):
                if d2_index>d1_index:
                    d2 = re.sub('[():\s]+', ' ', d2)
                    for s1_index, s1 in enumerate(d1.split("sentence_split")):
                        for s2_index, s2 in enumerate(d2.split("sentence_split")):
                            if len(s1.split())>2 and len(s2.split())>2:
                                sentence_pairs_cluster.append(str(sentence_pair_index) + "\t" + str(c_index) + "_" + str(d1_index) + "_" + str(s1_index) + "\t" + str(c_index) + "_" + str(d2_index) + "_" + str(s2_index) + "\t" + "n" + "\t" + str(c_index) + "\n")
                                sentence_pair_index += 1

            for s_index, s in enumerate(d1.split("sentence_split")):
                sentence_dict[str(c_index) + "_" + str(d1_index) + "_" + str(s_index)] = s

        indexes = range(len(sentence_pairs_cluster))
        target_indexes = random.sample(indexes, int(len(indexes)*0.1))
        sentence_pairs_tmp = []
        for index in target_indexes:
            sentence_pairs_tmp.append(sentence_pairs_cluster[index])
        with open("../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/test.tsv",
                  "a+") as f:
            f.writelines(sentence_pairs_tmp)

    # change sentence pair index
    with open("../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/test.tsv") as f:
        sentence_pairs_all = [line.split("\t") for line in f.readlines()]
    sentence_pairs_new = []
    for sp_index, sp in enumerate(sentence_pairs_all):
        sp[0] = str(sp_index)
        sentence_pairs_new.append("\t".join(sp))
    with open("../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/test.tsv", "w") as f:
        f.writelines(sentence_pairs_new)


    print("all sentences", len(sentence_dict))
    with open("../../utilized_models/sembert/SemBERT-master/glue_data/WCEP/sentence_dict.json", "w") as f:
        f.write(json.dumps(sentence_dict))

    # sentence_pair_generating(source_documents, "WCEP")
    del source_documents


    # with open("../../baselines/sembert/SemBERT-master/glue_data/MULTIXSCIENCE/test.tsv") as f:
    #     lines = f.readlines()
    # print(len(lines))