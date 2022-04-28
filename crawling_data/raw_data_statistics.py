#! -*- coding: utf-8 -*-
# Data verification

import json

def nips_verify(file_name):
    f = open(file_name)
    data = json.load(f)
    paper_count = len(data)
    print("All items", paper_count)

    count = 0
    avg_token_count_response = 0
    for paper in data:
        if paper.get("meta_review", "")!="" and "reviews" in paper.keys() and paper.get("pdf")!="" and "author_responses" in paper.keys() and paper.get("abstract", "")!="" and paper["author_responses"].get("pdf")!="" and paper["author_responses"].get("responses", "")!="":
            count += 1

            response = paper["author_responses"]["responses"]
            token_count_response = len(response.split())
            avg_token_count_response += token_count_response
    print("Valid items", count)
    print("avg_token_count_response", avg_token_count_response/count)

def iclr_verify(file_name):
    f = open(file_name)
    data = json.load(f)
    paper_count = len(data)
    print("All items", paper_count)

    valid_paper_count = 0
    all_token_count_fdecision = 0
    all_count_direct_reviews = 0
    all_token_count_direct_reviews = 0
    all_count_responses = 0
    all_token_count_responses = 0
    keys_sets = set([])
    for paper in data:
        if paper.get("link", "") != "" and paper.get("title", "") != "" and paper.get("authors", "") != "" and paper.get("abstract", "") != "" and "final_decision" in paper.keys() and paper.get("reviews_commments", []) != []:
            valid_paper_count += 1
            final_decision = paper["final_decision"]["content"]["comment"]# it should be metareview when in 2019
            # print(final_decision)
            token_count_fdecision = len(final_decision.split())
            all_token_count_fdecision += token_count_fdecision


            direct_reviews = []
            responses = []
            for rc in paper["reviews_commments"]:
                if rc["replyto"] == paper["id"]:# official review or public comment
                    cs = rc["content"].keys()
                    keys_sets.add(",".join(sorted(list(cs))))
                    if "review" in cs:
                        # print(rc)
                        direct_reviews.append(rc["content"]["review"])
                    elif "comment" in cs:
                        direct_reviews.append(rc["content"]["comment"])
                    else:
                        print(cs)
                else:
                    responses.append(rc["content"]["comment"])

            all_count_direct_reviews += len(direct_reviews)
            for d in direct_reviews:
                # print(len(d.split()))
                all_token_count_direct_reviews += len(d.split())

            all_count_responses += len(responses)
            for r in responses:
                all_token_count_responses += len(r.split())

    print("Valid papers", valid_paper_count)
    print("content key set", keys_sets)
    print("avg_token_count_fdecision", all_token_count_fdecision / valid_paper_count)
    print("avg_count_direct_reviews", all_count_direct_reviews / valid_paper_count)
    print("avg_token_count_direct_review", all_token_count_direct_reviews / all_count_direct_reviews)
    print("avg_count_responses", all_count_responses / valid_paper_count)
    print("avg_token_count_response", all_token_count_responses / all_count_responses)

def iclr_2017_verify():
    import os
    g = os.walk("iclr_2017/reviews_raw")

    avg_count_reviews = 0
    paper_count = 0
    paper_has_meta_review = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            f = open(os.path.join(path, file_name))
            data = json.load(f)
            print(data["accepted"])
            avg_count_reviews += len(data["reviews"])
            paper_count += 1

            print(len(data["reviews"]))
            print("-----")

            has_meta_review = 0
            for review in data["reviews"]:
                # print(review)
                if review["IS_META_REVIEW"] == True:
                    has_meta_review = 1
            paper_has_meta_review += has_meta_review

    print(paper_count)
    print(paper_has_meta_review)
    print(avg_count_reviews / paper_count)


if __name__ == "__main__":
    file_name = "data/nips_2019.json"
    nips_verify(file_name)

    file_name = "data/iclr_2020.json"
    iclr_verify(file_name)

    iclr_2017_verify()



