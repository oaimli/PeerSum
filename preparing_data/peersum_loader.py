# -*- coding: UTF-8 -*-
# Loading content is specified, not all information in peersum.


import json

def loading_all(folder, conference="all", data_name="peersum"):
    with open(folder + "/dataset/%s.json"%data_name, "r") as f:
        papers = json.load(f)

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


def loading_split(papers, including_public_comments=True, including_responses=False, sentence_split=False):
    source_clusters = []
    summaries = []
    paper_ids = []
    for paper in papers:
        summary = paper["meta_review"]
        paper_id_items = []
        for i, id_item in enumerate(paper["paper_id"].split("_")):
            if i > 1:
                paper_id_items.append(id_item)
        paper_id = "_".join(paper_id_items)
        # # this does not work because in the some paper ids there are underscore labels
        # paper_id = paper["paper_id"].split("_")[-1]
        reviews = []
        ratings = []
        has_direct_public = False
        has_direct_author = False
        for review in paper["reviews"]:
            text = review["content"]["comment"]
            if "replyto" in review.keys():
                if review["replyto"] == paper_id:
                    if review["writer"] == "author" and including_responses == True and text.strip() != "":
                        reviews.append(text)
                        has_direct_author = True
                    if review["writer"] == "public" and including_public_comments == True and text.strip() != "":
                        reviews.append(text)
                        has_direct_public = True
                    if review["writer"] == "official_reviewer" and text.strip() != "":
                        reviews.append(text)

                        content = review["content"]
                        if "rating" in content.keys():
                            ratings.append(content["rating"])
                else:
                    if including_responses == True and text.strip() != "":
                        reviews.append(text)
            else:
                # there are no responses for some conferences like nips 2019
                if review["writer"] == "author" and including_responses==True and text.strip()!="":
                    reviews.append(text)
                if review["writer"] == "public" and including_public_comments == True and text.strip() != "":
                    reviews.append(text)
                if review["writer"] == "official_reviewer" and text.strip() != "":
                    reviews.append(text)

        # # select some papers
        # lower_than = False
        # larger_than = False
        # for r in ratings:
        #     if int(r[0])>=6:
        #         larger_than=True
        #     if int(r[0])<=3:
        #         lower_than=True
        # if has_direct_public and has_direct_author and lower_than and larger_than:
        #     print(paper["title"])

        if summary.strip() != "" and len(reviews) > 0:
            if not sentence_split:
                summary = " ".join(summary.split("sentence_split"))
                for i, review in enumerate(reviews):
                    reviews[i] = " ".join(review.split("sentence_split"))
            source_clusters.append(reviews)
            summaries.append(summary)
            paper_ids.append(paper["paper_id"])
        else:
            print(paper)
            break
    return source_clusters, summaries, paper_ids


def loading_peersum(folder, including_public_comments=True, including_responses=False, data_name="peersum", spliting=False, sentence_split=False, with_id=False):
    print("loading", data_name)
    with open(folder + "/dataset/%s.json"%data_name, "r") as f:
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

    source_clusters_train, summaries_train, paper_ids_train = loading_split(papers_train, including_public_comments, including_responses, sentence_split)
    source_clusters_val, summaries_val, paper_ids_val = loading_split(papers_val, including_public_comments, including_responses, sentence_split)
    source_clusters_test, summaries_test, paper_ids_test = loading_split(papers_test, including_public_comments, including_responses, sentence_split)

    if with_id:
        return paper_ids_train, paper_ids_val, paper_ids_test

    print("all", len(papers))
    print("train samples", len(papers_train), len(source_clusters_train), len(summaries_train))
    print("val samples", len(papers_val), len(source_clusters_val), len(summaries_val))
    print("test samples", len(papers_test), len(source_clusters_test), len(summaries_test))

    if spliting:
        return source_clusters_train, summaries_train, source_clusters_val, summaries_val, source_clusters_test, summaries_test
        # return source_clusters_train[:100], summaries_train[:100], source_clusters_val[:100], summaries_val[:100], source_clusters_test[:100], summaries_test[:100]# for debugging
    else:
        return source_clusters_train+source_clusters_val+source_clusters_test, summaries_train+summaries_val+summaries_test


if __name__ == "__main__":
    import random
    source_clusters, summaries= loading_peersum("../../peersum", data_name="peersum_new_cleaned", including_public_comments=False, including_responses=False, spliting=False)
    print("source_documents", len(source_clusters))
    print("summaries", len(summaries))


    indexes = range(len(summaries))
    target_indexes = random.sample(indexes, 2)
    for i in target_indexes:
        print("*********************")
        for item in source_clusters[i]:
            print(item)
            print("###########")
        print(summaries[i])