# -*- coding: UTF-8 -*-
# overview statistics of PeerSum

import sys
import numpy as np
import math

sys.path.append("../../")
from peersum.preparing_data.peersum_loader import loading_all


papers = loading_all("../../peersum", "all", "peersum")

official_review_texts = [] # the first level comment by official reviewers
public_comment_texts = [] # the first level comment by public readers
response_texts = [] # responses from authors and discussions with readers
meta_review_texts = []

avg_rating = []
avg_rating_variance = []
avg_condifence = []
avg_confidence_variance = []
rating_keys = set([])
confidence_keys = set([])

all_document_pairs_with_rating = 0
contradictory_document_pairs = 0
controversial_papers = 0
papers_with_rating = 0


for paper in papers:
    ratings = []
    confidences = []

    meta_review_texts.append(paper["meta_review"])

    paper_id_items = []
    for i, id_item in enumerate(paper["paper_id"].split("_")):
        if i>1:
            paper_id_items.append(id_item)

    paper_id = "_".join(paper_id_items)

    for r in paper["reviews"]:
        c = r["content"]

        if "rating" in c.keys():
            if isinstance(c["rating"], int):
                ratings.append(c["rating"])
            else:
                ratings.append(int(c["rating"].split(":")[0].strip()))
            rating_keys.add(c["rating"])
        if "confidence" in c.keys():
            confidence_keys.add(c["confidence"])
            if isinstance(c["confidence"], int):
                confidences.append(c["confidence"])
            else:
                confidences.append(int(c["confidence"].split(":")[0].strip()))

        if "replyto" in r.keys():
            if r["replyto"] == paper_id:
                if r["writer"]=="official_reviewer":
                    official_review_texts.append(c["comment"])
                elif r["writer"] == "public":
                    public_comment_texts.append(c["comment"])
                else:
                    response_texts.append(c["comment"])
            else:
                response_texts.append(c["comment"])
        else:
            if r["writer"]=="official_reviewer":
                official_review_texts.append(c["comment"])
            elif r["writer"] == "public":
                public_comment_texts.append(c["comment"])
            elif r["writer"] == "author":
                response_texts.append(c["comment"])
            else:
                print("something wrong")
    if len(ratings)>0:
        papers_with_rating += 1
        avg_rating.append(np.mean(ratings))
        avg_rating_variance.append(np.var(ratings))

        has_conflict = False
        for rating1_index, rating1 in enumerate(ratings):
            for rating2_index, rating2, in enumerate(ratings):
                all_document_pairs_with_rating += 1
                if abs(rating2 - rating1)>=5:
                    contradictory_document_pairs += 1
                    has_conflict = True
        if has_conflict==True:
            controversial_papers += 1


    if len(confidences)>0:
        avg_condifence.append(np.mean(confidences))
        avg_confidence_variance.append(np.var(confidences))


print("Official reviews per paper:", len(official_review_texts)/len(papers))
print("Public comments per paper:", len(public_comment_texts)/len(papers))
print("Responses per paper:", len(response_texts)/len(papers))
print("Ratings", rating_keys)
print("Avg rating per official review", np.mean(avg_rating))
print("Rating variance for a paper", np.mean(avg_rating_variance))
print("Confidences", confidence_keys)
print("Avg confidence per official review", np.mean(avg_condifence))
print("Confidence variance for a paper", np.mean(avg_confidence_variance))

print("Percentage of contradictory review pairs", float(contradictory_document_pairs)/float(all_document_pairs_with_rating))
print("controversial papers", float(controversial_papers)/float(papers_with_rating))


# nlp = spacy.load("en_core_web_sm")
# all_word_count_mreview = []
# all_sent_count_mreview = []
# for mreview in meta_review_texts:
#     sents = mreview.split("sentence_split")
#     all_sent_count_mreview.append(len(sents))
#     all_word_count_mreview.append(len(" ".join(sents).split()))
# if len(meta_review_texts)>0:
#     print("Avg tokens per meta review", np.mean(all_word_count_mreview))
#     print("Avg sents per meta review", np.mean(all_sent_count_mreview))
# else:
#     print("Avg tokens per meta review", 0)
#
# all_word_count_oreview = []
# all_sent_count_oreview = []
# for oreview in official_review_texts:
#     sents = oreview.split("sentence_split")
#     all_sent_count_oreview.append(len(sents))
#     all_word_count_oreview.append(len(" ".join(sents).split()))
# if len(official_review_texts)>0:
#     print("Avg tokens per official review", np.mean(all_word_count_oreview))
#     print("Avg sents per official review", np.mean(all_sent_count_oreview))
# else:
#     print("Avg tokens per official review", 0)
#
# all_word_count_pcomment = []
# all_sent_count_pcomment = []
# for pcomment in public_comment_texts:
#     sents = pcomment.split("sentence_split")
#     all_sent_count_pcomment.append(len(sents))
#     all_word_count_pcomment.append(len(" ".join(sents).split()))
# if len(public_comment_texts)>0:
#     print("Avg tokens per public comment", np.mean(all_word_count_pcomment))
#     print("Avg sents per public comment", np.mean(all_sent_count_pcomment))
# else:
#     print("Avg tokens per public comment", 0)
#
# all_word_count_response = []
# all_sent_count_response = []
# for response in response_texts:
#     sents = response.split("sentence_split")
#     all_sent_count_response.append(len(sents))
#     all_word_count_response.append(len(" ".join(sents).split()))
# if len(response_texts)>0:
#     print("Avg tokens per response", np.mean(all_word_count_response))
#     print("Avg sents per response", np.mean(all_sent_count_response))
# else:
#     print("Avg tokens per response", 0)