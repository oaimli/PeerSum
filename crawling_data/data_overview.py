# -*- coding: UTF-8 -*-
# overview statistics of all data

import sys
import numpy as np
import math
import random
import jsonlines


def loading_original(conference="all", data_name="peersum_all"):
    papers = []
    with jsonlines.open("data/%s.json"%data_name, "r") as reader:
        for paper in reader:
            papers.append(paper)

    papers_iclr17 = []
    papers_iclr18 = []
    papers_iclr19 = []
    papers_iclr20 = []
    papers_iclr21 = []
    papers_iclr22 = []
    papers_nips19 = []
    papers_nips20 = []
    papers_nips21 = []
    for paper in papers:
        if "iclr_2017" in paper["paper_id"]:
            papers_iclr17.append(paper)
        if "iclr_2018" in paper["paper_id"]:
            papers_iclr18.append(paper)
        if "iclr_2019" in paper["paper_id"]:
            papers_iclr19.append(paper)
        if "iclr_2020" in paper["paper_id"]:
            papers_iclr20.append(paper)
        if "iclr_2021" in paper["paper_id"]:
            papers_iclr21.append(paper)
        if "iclr_2022" in paper["paper_id"]:
            papers_iclr22.append(paper)
        if "nips_2019" in paper["paper_id"]:
            papers_nips19.append(paper)
        if "nips_2020" in paper["paper_id"]:
            papers_nips20.append(paper)
        if "nips_2021" in paper["paper_id"]:
            papers_nips21.append(paper)

    if conference== "all":
        print("Papers:", len(papers))
        return papers
    elif conference== "nips":
        papers_nips = papers_nips19 + papers_nips20 + papers_nips21
        print("Papers NIPS", len(papers_nips))
        return papers_nips
    elif conference== "iclr":
        papers_iclr = papers_iclr17 + papers_iclr18 + papers_iclr19 + papers_iclr20 + papers_iclr21 + papers_iclr22
        print("Papers ICLR", len(papers_iclr))
        return papers_iclr
    else:
        print("Not a right conference parameter")

papers = loading_original("iclr", "peersum_all")

nips = []
iclr = []
for paper in papers:
    if "iclr" in paper["paper_id"]:
        iclr.append(paper)
    if "nips" in paper["paper_id"]:
        nips.append(paper)

print("nips", len(nips))
print("iclr", len(iclr))

official_threads = []
public_threads = []
author_threads = []
meta_review_texts = []
paper_scores = []

for paper in papers:
    meta_review_texts.append(paper["meta_review"])
    paper_score = paper['paper_score']
    if paper_score!=-1:
        paper_scores.append(paper_score)

    official_threads.append(len(paper["official_threads"]))
    public_threads.append(len(paper["public_threads"]))
    author_threads.append(len(paper["author_threads"]))


print("The number of official threads each paper", np.mean(official_threads))
print("The number of public threads each paper", np.mean(public_threads))
print("The number of author threads each paper", np.mean(author_threads))
print("Avg score", np.mean(paper_scores))