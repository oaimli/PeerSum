import sys
sys.path.append('../')
import numpy as np
import os
import random
import json

from ref_free_metrics.supert import Supert


def loading_split(papers, including_public_comments=True, including_responses=False):
    source_clusters = []
    summaries = []
    for paper in papers:
        summary = paper["meta_review"]
        paper_id_items = []
        for i, id_item in enumerate(paper["paper_id"].split("_")):
            if i > 1:
                paper_id_items.append(id_item)

        paper_id = "_".join(paper_id_items)
        # paper_id = paper["paper_id"].split("_")[-1]
        reviews = []
        for review in paper["reviews"]:
            text = review["content"]["comment"]
            if "replyto" in review.keys():
                if review["replyto"] == paper_id:
                    if review["writer"] == "author" and including_responses == True and text.strip() != "":
                        reviews.append(text)
                    if review["writer"] == "public" and including_public_comments == True and text.strip() != "":
                        reviews.append(text)
                    if review["writer"] == "official_reviewer" and text.strip() != "":
                        reviews.append(text)
                else:
                    if including_responses == True and text.strip() != "":
                        reviews.append(text)
            else:
                if review["writer"] == "author" and including_responses == True and text.strip() != "":
                    reviews.append(text)
                if review["writer"] == "public" and including_public_comments == True and text.strip() != "":
                    reviews.append(text)
                if review["writer"] == "official_reviewer" and text.strip() != "":
                    reviews.append(text)

        if summary.strip() != "" and len(reviews) > 0:
            source_clusters.append(reviews)
            summaries.append(summary)
        if len(reviews)==0:
            print(paper)
    return source_clusters, summaries


def loading_peersum(folder, including_public_comments=True, including_responses=False, data_name="peersum", spliting=False):
    print("loading", data_name)
    with open(folder + "peersum/data/%s.json"%data_name, "r") as f:
        papers = json.load(f)

    # indexes = range(len(papers))
    # target_indexes = random.sample(indexes, 5)
    # for i in target_indexes:
    #     print(papers[i])

    papers_train = []
    papers_val = []
    papers_test = []
    for paper in papers:
        if paper["label"]=="train":
            papers_train.append(paper)
        if paper["label"]=="val":
            papers_val.append(paper)
        if paper["label"]=="test":
            papers_test.append(paper)

    source_clusters_train, summaries_train = loading_split(papers_train, including_public_comments, including_responses)
    source_clusters_val, summaries_val = loading_split(papers_val, including_public_comments, including_responses)
    source_clusters_test, summaries_test = loading_split(papers_test, including_public_comments, including_responses)

    print("all", len(papers))
    print("train samples", len(papers_train), len(source_clusters_train), len(summaries_train))
    print("val samples", len(papers_val), len(source_clusters_val), len(summaries_val))
    print("test samples", len(papers_test), len(source_clusters_test), len(summaries_test))

    if spliting:
        return source_clusters_train, summaries_train, source_clusters_val, summaries_val, source_clusters_test, summaries_test
        # return source_clusters_train[:100], summaries_train[:100], source_clusters_val[:100], summaries_val[:100], source_clusters_test[:100], summaries_test[:100]# for debugging
    else:
        return source_clusters_train+source_clusters_val+source_clusters_test, summaries_train+summaries_val+summaries_test


def transform_data(source_clusters, summaries, count=100):
    count_all = len(source_clusters)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, count)
    source_clusters_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_clusters_tmp.append(source_clusters[i])
        summaries_tmp.append(summaries[i])

    source_docs = []
    for docs_index, docs in enumerate(source_clusters_tmp):
        for doc_index, doc in enumerate(docs):
            sents = doc.split("sentence_split")
            source_docs.append((str(docs_index)+"."+str(doc_index), sents))

    return source_docs, summaries_tmp

def loading_mds(folder, data_name, spliting=True):
    print("loading mds", data_name)

    samples = []
    file_name = folder + "processed/%s.json"%data_name
    if os.path.isfile(file_name):
        with open(file_name, "r") as f:
            samples = json.load(f)
    else:
        print("please do pre pre_processing")

    source_clusters_train = []
    summaries_train = []
    source_clusters_val = []
    summaries_val = []
    source_clusters_test = []
    summaries_test = []
    for sample in samples:
        summary = sample["summary"]
        source_documents = sample["source_documents"]
        if sample["label"] == "train" and summary.strip()!="" and len(source_documents)>0:
            source_clusters_train.append(source_documents)
            summaries_train.append(summary)
        if sample["label"] == "val" and summary.strip()!="" and len(source_documents)>0:
            source_clusters_val.append(source_documents)
            summaries_val.append(summary)
        if sample["label"] == "test" and summary.strip()!="" and len(source_documents)>0:
            source_clusters_test.append(source_documents)
            summaries_test.append(summary)

    print("all", len(summaries_train + summaries_val + summaries_test))
    print("train samples", len(source_clusters_train))
    print("val samples", len(source_clusters_val))
    print("test samples", len(source_clusters_test))

    if spliting:
        return source_clusters_train, summaries_train, source_clusters_val, summaries_val, source_clusters_test, summaries_test
        # return source_clusters_train[:100], summaries_train[:100], source_clusters_val[:100], summaries_val[:100], source_clusters_test[:100], summaries_test[:100]  # for debugging
    else:
        return source_clusters_train+source_clusters_val+source_clusters_test, summaries_train+summaries_val+summaries_test


if __name__ == '__main__':
    # pseudo-ref strategy: 
    # * top15 means the first 15 sentences from each input doc will be used to build the pseudo reference summary
    pseudo_ref = 'top15'

    sampling_count = 200

    # PeerSum with only official reviews
    source_clusters, summaries = loading_peersum("../../../../peersum", data_name="peersum_cleaned",
                                                 including_public_comments=False, including_responses=False,
                                                 spliting=False)
    source_docs, summaries = transform_data(source_clusters, summaries, sampling_count)

    # get unsupervised metrics for the summaries
    supert = Supert(source_docs, ref_metric=pseudo_ref)
    scores = supert(summaries)
    print('Relevance of summaries for PeerSum-R\n', np.mean(scores), np.var(scores))


    # PeerSum including comments
    source_clusters, summaries = loading_peersum("../../../../peersum", data_name="peersum_cleaned",
                                                 including_public_comments=True, including_responses=False,
                                                 spliting=False)
    source_docs, summaries = transform_data(source_clusters, summaries, sampling_count)

    # get unsupervised metrics for the summaries
    supert = Supert(source_docs, ref_metric=pseudo_ref)
    scores = supert(summaries)
    print('Relevance of summaries for PeerSum-RC\n', np.mean(scores), np.var(scores))


    # PeerSum including comments and respoonses
    source_clusters, summaries = loading_peersum("../../../../peersum", data_name="peersum_cleaned", including_public_comments=True, including_responses=True, spliting=False)
    source_docs, summaries = transform_data(source_clusters, summaries, sampling_count)

    # get unsupervised metrics for the summaries
    supert = Supert(source_docs, ref_metric=pseudo_ref)
    scores = supert(summaries)
    print('Relevance of summaries for PeerSum-ALL\n', np.mean(scores), np.var(scores))



    for data_name in ["multinews", "multixscience", "wikisum", "wcep"]:
        # source_clusters, summaries = loading_mds(folder="../../../../preparing_data/", data_name="%s_cleaned"%data_name, spliting=False)
        source_documents, summaries = loading_mds(folder="/scratch/miao4/datasets_tmp/", data_name="%s_cleaned"%data_name, spliting=False)
        # print(source_clusters[0])
        source_docs, summaries = transform_data(source_clusters, summaries, sampling_count)

        # get unsupervised metrics for the summaries
        supert = Supert(source_docs, ref_metric=pseudo_ref)
        scores = supert(summaries)
        print('Relevance of summaries for %s\n'%data_name, np.mean(scores), np.var(scores))




