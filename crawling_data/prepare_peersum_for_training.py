# Transform PeerSum dataset into training datasets for other baseline models like BART


# -*- coding: UTF-8 -*-
import random

import jsonlines
import numpy as np

def loading_peersum(folder="data", including_public=True, including_author=True, including_abstract=True):
    print("loading peersum")
    papers = []
    with jsonlines.open(folder + "/peersum_all.json", "r") as reader:
        for paper in reader:
            papers.append(paper)
    clusters = []
    for paper in papers:
        summary = paper["meta_review"]
        paper_id = paper['paper_id']
        source_documents = []
        if including_abstract:
            source_documents.append(paper["paper_abstract"])

        for review in paper["reviews"]:
            if "comment" in review.keys():
                text = review["comment"]
                if review["writer"] == "official_reviewer":
                    source_documents.append(text)
                if review["writer"] == "author" and including_author == True:
                    source_documents.append(text)
                if review["writer"] == "public" and including_public == True:
                    source_documents.append(text)

        if summary.strip() != "" and len(source_documents) > 1:
            random.shuffle(source_documents)
            clusters.append({"source_documents": source_documents, "summary": summary, "paper_id": paper_id, "label": paper["label"], "paper_acceptance": paper["paper_acceptance"]})
        else:
            print(paper["paper_id"])

    return clusters


def prepare_peersum_with_disagreements(folder="data"):
    print("loading peersum")
    papers = []
    with jsonlines.open(folder + "/peersum_all.json", "r") as reader:
        for paper in reader:
            papers.append(paper)
    clusters = []
    for paper in papers:
        summary = paper["meta_review"]
        paper_id = paper['paper_id']
        source_documents = []

        with_disagreements = False
        for i, review_i in enumerate(paper["reviews"]):
            for j, review_j in enumerate(paper["reviews"]):
                if j > i:
                    if review_i["rating"] > 0 and review_j["rating"] > 0:
                        dis = review_i["rating"] - review_j["rating"]
                        if abs(dis) >= 5:
                            with_disagreements = True

        source_documents.append(paper["paper_abstract"])
        for review in paper["reviews"]:
            if "comment" in review.keys():
                text = review["comment"]
                source_documents.append(text)

        if summary.strip() != "" and len(source_documents) > 1 and with_disagreements==True:
            random.shuffle(source_documents)
            clusters.append({"source_documents": source_documents, "summary": summary, "paper_id": paper_id, "label": paper["label"]})
        # else:
        #     print(paper["paper_id"])

    print("all samples with disagreements", len(clusters))
    # with jsonlines.open("../../datasets/peersum_with_disagreements.json", "w") as writer:
    #     writer.write_all(clusters)


def prepare_peersum():
    """
    transform peersum into the format for multi-document summarization
    """
    samples = loading_peersum(including_public=True, including_author=True, including_abstract=True)
    with jsonlines.open("../../datasets/peersum.json", "w") as writer:
        writer.write_all(samples)


if __name__ == "__main__":
    prepare_peersum()
    prepare_peersum_with_disagreements()

    import random
    papers = loading_peersum(including_public=True, including_author=True, including_abstract=True)
    print("all samples", len(papers))
    indexes = range(len(papers))
    target_indexes = random.sample(indexes, 2)
    for i in target_indexes:
        paper = papers[i]
        print("*********************")
        print(paper["paper_id"])
        print("*")
        for item in paper["source_documents"]:
            print(item)
            print("###########")
        print(paper["summary"])