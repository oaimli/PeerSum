# -*- coding: UTF-8 -*-
import random

import jsonlines
import numpy as np

def loading_peersum(folder="../crawling_data", including_public=True, including_author=True, including_abstract=True):
    print("loading peersum")
    papers = []
    with jsonlines.open(folder + "/data/peersum_all.json", "r") as reader:
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
