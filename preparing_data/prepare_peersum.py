# peersum dataset constructing, transforming crawled raw data into peersum

import json
import os
import time
import random
import openreview
def split_data(papers):
    all_num = len(papers)
    train_num = int(all_num * 0.8)
    val_num = int(all_num * 0.1)

    papers_indexes = range(all_num)
    papers_train_indexes = random.sample(papers_indexes, train_num)
    for i in papers_train_indexes:
        papers[i]["label"] = "train"
    print("train", len(papers_train_indexes))

    papers_train_test_indexes = [item for item in papers_indexes if item not in papers_train_indexes]
    papers_val_indexes = random.sample(papers_train_test_indexes, val_num)
    for i in papers_val_indexes:
        papers[i]["label"] = "val"
    print("val", len(papers_val_indexes))

    papers_test_indexes = [item for item in papers_train_test_indexes if item not in papers_val_indexes]
    for i in papers_test_indexes:
        papers[i]["label"] = "test"
    print("test", len(papers_test_indexes))


def prepare_iclr(year):
    # iclr preparing
    data_folder_iclr = "../crawling_data/data/"
    if int(year)==2017:
        # iclr_2017 pre_processing
        iclr_2017_papers = []
        g = os.walk(data_folder_iclr + "iclr_2017/reviews_raw")
        for path, dir_list, file_list in g:
            for file_name in file_list:
                paper_new = {}
                f = open(os.path.join(path, file_name))
                data = json.load(f)
                paper_new["paper_id"] = "iclr_2017_%s" % file_name.split(".")[0]
                paper_new["title"] = data["title"]
                paper_new["abstract"] = data["abstract"]
                if data["accepted"] == True:
                    paper_new["acceptance"] = "accepted"
                else:
                    paper_new["acceptance"] = "rejected"
                if "SCORE" in data.keys():
                    paper_new["score"] = data["SCORE"]
                    paper_new["meta_review"] = ""
                    reviews = []
                    for review in data["reviews"]:
                        if str(review["comments"]).strip() != "":
                            # print(review.keys())
                            if review["IS_META_REVIEW"] == True:
                                # print(review["IS_META_REVIEW"])
                                paper_new["meta_review"] = review["comments"]
                            else:
                                if "REVIEWER_CONFIDENCE" in review.keys():
                                    r = {}
                                    r["writer"] = "official_reviewer"
                                    content = {}
                                    content["comment"] = review["comments"]
                                    content["rating"] = review["RECOMMENDATION"]
                                    content["confidence"] = review["REVIEWER_CONFIDENCE"]
                                    r["content"] = content
                                else:
                                    r = {}
                                    r["writer"] = "public"
                                    content = {}
                                    content["comment"] = review["comments"]
                                    r["content"] = content
                                reviews.append(r)
                    paper_new["reviews"] = reviews

                    if paper_new["meta_review"] != "" and len(paper_new["reviews"]) > 0:
                        iclr_2017_papers.append(paper_new)
        split_data(iclr_2017_papers)
        return iclr_2017_papers

    if 2018<=int(year)<=2021:
        papers_iclr = []
        invitations = set([])
        with open(data_folder_iclr + "iclr_%s.json" % year) as f:
            paper_list = json.load(f)
            for paper in paper_list:
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              "") != "" and paper.get(
                        "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                                 []) != []:
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["title"] = paper["title"]
                    paper_new["abstract"] = paper["abstract"]
                    paper_new["acceptance"] = paper["comment"]
                    final_decision_content = paper["final_decision"]["content"]
                    final_decision_time = paper["final_decision"]["tmdate"]
                    if year == "2019":
                        final_decision = final_decision_content["metareview"]# it should be metareview when in 2019
                    else:
                        final_decision = final_decision_content["comment"]
                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                            if rc["invitation"].split("/")[-1] == "Comment":
                                print(paper["title"])

                        reviews = []
                        for rc in paper["reviews_commments"]:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tmdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]

                                invit = rc["invitation"].split("/")[-1]
                                if "Official_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"

                                review["replyto"] = rc["replyto"]

                                content = {}
                                cs = rc["content"]
                                if "review" in cs.keys():
                                    content["comment"] = cs["review"]
                                if "comment" in cs.keys():
                                    content["comment"] = cs["comment"]
                                if "rating" in cs.keys():
                                    content["rating"] = cs["rating"]
                                if "confidence" in cs.keys():
                                    content["confidence"] = cs["confidence"]
                                review["content"] = content
                                if len(content.keys()) > 0:
                                    reviews.append(review)
                        paper_new["reviews"] = reviews
                        if len(reviews) > 0:
                            papers_iclr.append(paper_new)
            # print(sorted(invitations))

        split_data(papers_iclr)
        return papers_iclr

    if int(year)==2022:
        papers_iclr = []
        invitations = set([])
        with open(data_folder_iclr + "iclr_%s.json" % year) as f:
            paper_list = json.load(f)
            print("paper count", len(paper_list))
            for paper in paper_list:
                # if paper["title"] == "Decoupled Contrastive Learning":
                #     print(paper)
                #     break
                paper_new = {}
                if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                              []) != [] and paper.get(
                    "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                             []) != []:
                    paper_new["paper_id"] = "iclr_" + year + "_" + paper["id"]
                    paper_new["title"] = paper["title"]
                    paper_new["abstract"] = paper["abstract"]
                    paper_new["acceptance"] = paper["comment"]
                    final_decision_content = paper["final_decision"]["content"]
                    final_decision_time = paper["final_decision"]["tmdate"]
                    final_decision = final_decision_content["comment"]
                    if final_decision != "":
                        paper_new["meta_review"] = final_decision

                        # if y == "2020":
                        for rc in paper["reviews_commments"]:
                            # if rc["replyto"] == paper["id"]:
                            for s in rc["signatures"]:
                                invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                            if rc["invitation"].split("/")[-1] == "Comment":
                                print(paper["title"])

                        reviews = []
                        for rc in paper["reviews_commments"]:
                            if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tmdate"] / 1000):
                                review = {}
                                review["review_id"] = rc["id"]

                                invit = rc["invitation"].split("/")[-1]
                                if "Official_Review" == invit:
                                    review["writer"] = "official_reviewer"
                                elif "Official_Comment" == invit or "Comment" == invit:
                                    signatures = " ".join(rc["signatures"])
                                    if "Authors" in signatures:
                                        review["writer"] = "author"
                                    elif "AnonReviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Reviewer" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Area_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    elif "Program_Chair" in signatures:
                                        review["writer"] = "official_reviewer"
                                    else:
                                        review["writer"] = "public"

                                elif "Public_Comment" == invit:
                                    review["writer"] = "public"
                                else:
                                    signatures = " ".join(rc["signatures"])
                                    if "Author" in signatures:
                                        review["writer"] = "author"
                                    else:
                                        review["writer"] = "public"

                                review["replyto"] = rc["replyto"]

                                content = {}
                                cs = rc["content"]
                                review_text = ""
                                if "review" in cs.keys():
                                    review_text = review_text + " " + cs["review"]
                                if "comment" in cs.keys():
                                    review_text = review_text + " " + cs["comment"]
                                if "main_review" in cs.keys():
                                    review_text = cs["main_review"]
                                if "summary_of_the_review" in cs.keys():
                                    review_text = review_text + " " + cs["summary_of_the_review"]
                                content["comment"] = review_text
                                if "rating" in cs.keys():
                                    content["rating"] = cs["rating"]
                                if "recommendation" in cs.keys():
                                    content["rating"] = cs["recommendation"]
                                if "confidence" in cs.keys():
                                    content["confidence"] = cs["confidence"]
                                review["content"] = content
                                if len(content.keys()) > 0:
                                    reviews.append(review)
                        paper_new["reviews"] = reviews
                        if len(reviews) > 0:
                            papers_iclr.append(paper_new)
                # else:
                #     print(paper["title"])
            # print(sorted(invitations))

        split_data(papers_iclr)
        return papers_iclr
    return []


def prepare_nips(year):
    data_folder_nips = "../crawling_data/data/"
    # nips pre_processing
    if 2019<=int(year)<=2020:
        papers_nips = []
        with open(data_folder_nips + "nips_%s.json" % year) as f:
            paper_list = json.load(f)
            for paper in paper_list:
                paper_new = {}
                if paper.get("meta_review", "").strip() != "" and "reviews" in paper.keys() and paper.get(
                        "pdf") != "" and "author_responses" in paper.keys() and paper.get("abstract", "") != "" and \
                        paper[
                            "author_responses"].get("pdf") != "" and paper["author_responses"].get("responses",
                                                                                                   "") != "":
                    paper_new["paper_id"] = "nips_" + year + "_" + paper["link"].split("/")[-1].split("-")[0]
                    paper_new["title"] = paper["title"]
                    paper_new["abstract"] = paper["abstract"]
                    paper_new["meta_review"] = paper["meta_review"]
                    paper_new["acceptance"] = paper["comment"]

                    # parse the response?
                    reviews = []
                    if year == "2019":
                        for review in paper["reviews"]:
                            r = {}
                            r["writer"] = "official_reviewer"
                            r["content"] = {"comment": review}
                            reviews.append(r)
                    else:
                        for d in paper["reviews"]:
                            tmp = ""
                            for r in d.keys():
                                rv = r + d[r]
                                tmp += rv
                            r = {}
                            r["writer"] = "official_reviewer"
                            r["content"] = {"comment": tmp}
                            reviews.append(r)
                    paper_new["reviews"] = reviews

                    papers_nips.append(paper_new)
        split_data(papers_nips)
        return papers_nips

    if int(year)==2021:
        papers_nips = []
        invitations = set([])
        with open(data_folder_nips + "nips_%s.json" % year) as f:
            paper_list = json.load(f)
        print("paper count", len(paper_list))
        for paper in paper_list:
            paper_new = {}

            if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors",
                                                                                          "") != "" and paper.get(
                "abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments",
                                                                                         []) != []:
                paper_new["paper_id"] = "nips_" + year + "_" + paper["id"]
                paper_new["title"] = paper["title"]
                paper_new["abstract"] = paper["abstract"]
                paper_new["acceptance"] = paper["comment"]
                final_decision_content = paper["final_decision"]["content"]
                final_decision_time = paper["final_decision"]["tmdate"]
                final_decision = final_decision_content["comment"]
                if final_decision != "":
                    paper_new["meta_review"] = final_decision

                    # if y == "2020":
                    for rc in paper["reviews_commments"]:
                        # if rc["replyto"] == paper["id"]:
                        for s in rc["signatures"]:
                            invitations.add(rc["invitation"].split("/")[-1] + "#" + s.split("/")[-1])
                        if rc["invitation"].split("/")[-1] == "Comment":
                            print(paper["title"])

                    reviews = []
                    for rc in paper["reviews_commments"]:
                        if time.localtime(final_decision_time / 1000) >= time.localtime(rc["tmdate"] / 1000):
                            review = {}
                            review["review_id"] = rc["id"]

                            invit = rc["invitation"].split("/")[-1]
                            if "Official_Review" == invit:
                                review["writer"] = "official_reviewer"
                            elif "Official_Comment" == invit or "Comment" == invit:
                                signatures = " ".join(rc["signatures"])
                                if "Authors" in signatures:
                                    review["writer"] = "author"
                                elif "AnonReviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Reviewer" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Area_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                elif "Program_Chair" in signatures:
                                    review["writer"] = "official_reviewer"
                                else:
                                    review["writer"] = "public"

                            elif "Public_Comment" == invit:
                                review["writer"] = "public"
                            else:
                                signatures = " ".join(rc["signatures"])
                                if "Author" in signatures:
                                    review["writer"] = "author"
                                else:
                                    review["writer"] = "public"

                            review["replyto"] = rc["replyto"]

                            content = {}
                            cs = rc["content"]
                            review_text = ""
                            if "review" in cs.keys():
                                review_text = review_text + " " + cs["review"]
                            if "comment" in cs.keys():
                                review_text = review_text + " " + cs["comment"]
                            if "main_review" in cs.keys():
                                review_text = cs["main_review"]
                            if "summary_of_the_review" in cs.keys():
                                review_text = review_text + " " + cs["summary_of_the_review"]
                            content["comment"] = review_text
                            if "rating" in cs.keys():
                                content["rating"] = cs["rating"]
                            if "recommendation" in cs.keys():
                                content["rating"] = cs["recommendation"]
                            if "confidence" in cs.keys():
                                content["confidence"] = cs["confidence"]
                            review["content"] = content
                            if len(content.keys()) > 0:
                                reviews.append(review)
                    paper_new["reviews"] = reviews
                    if len(reviews) > 0:
                        papers_nips.append(paper_new)
            # else:
            #     print(paper["title"])
        # print(sorted(invitations))
        split_data(papers_nips)
        return papers_nips



if __name__ == "__main__":
    # # get a sample with forum id
    # base_url = "https://api.openreview.net"
    # client = openreview.Client(baseurl=base_url)
    # notes = client.get_notes(
    #     forum="JzdYX8uzT4W")
    # for note in notes:
    #     print(note)

    papers_iclr_2022 = prepare_iclr("2022")
    print("ICLR 2022", len(papers_iclr_2022))
    papers_nips_2021 = prepare_nips("2021")
    print("NIPS 2021", len(papers_nips_2021))

    # # sampling to check
    # indexes = random.sample(range(len(papers_nips_2021)), 1)
    # for index in indexes:
    #     print(papers_nips_2021[index])
    # indexes = random.sample(range(len(papers_iclr_2022)), 1)
    # for index in indexes:
    #     print(papers_iclr_2022[index])

    peersum_file = "../dataset/peersum.json"
    if os.path.exists(peersum_file):
        with open(peersum_file, "r") as f:
            papers = json.load(f)
            print("papers exist", len(papers))
    else:
        papers = []

    papers.extend(papers_iclr_2022)
    papers.extend(papers_nips_2021)


    papers_train = []
    papers_val = []
    papers_test = []
    for paper in papers:
        if paper["label"] == "train":
            papers_train.append(paper)
        if paper["label"] == "val":
            papers_val.append(paper)
        if paper["label"] == "test":
            papers_test.append(paper)
    print("train all", len(papers_train))
    print("val all", len(papers_val))
    print("test all", len(papers_test))

    with open("../dataset/peersum_new.json", "w") as f:
        f.write(json.dumps(papers))


