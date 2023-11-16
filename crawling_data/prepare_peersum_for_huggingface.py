# -*- coding: UTF-8 -*-
# Transform PeerSum dataset into training datasets for other baseline models like BART

import random
import jsonlines


def prepare_for_huggingface(folder="data"):
    print("loading peersum")
    papers = []
    with jsonlines.open(folder + "/peersum_all.json", "r") as reader:
        for paper in reader:
            papers.append(paper)

    samples = []
    for paper in papers:
        review_ids = []
        review_writers =[]
        review_contents = []
        review_ratings = []
        review_confidences = []
        review_reply_tos = []
        for review in paper["reviews"]:
            review_ids.append(review["review_id"])
            review_writers.append(review["writer"])
            review_contents.append(review["comment"])
            review_ratings.append(review["rating"])
            review_confidences.append(review["confidence"])
            review_reply_tos.append(review["reply_to"])

        paper["review_ids"] = review_ids
        paper["review_writers"] = review_writers
        paper["review_contents"] = review_contents
        paper["review_ratings"] = review_ratings
        paper["review_confidences"] = review_confidences
        paper["review_reply_tos"] = review_reply_tos

        del paper["reviews"]
        samples.append(paper)
    print(len(samples))
    with jsonlines.open("peersum_huggingface.jsonl", "w") as writer:
        writer.write_all(samples)



if __name__ == "__main__":
    prepare_for_huggingface()