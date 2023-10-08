# Sampling data from the original PeerSum dataset based on variance of review scores for individual papers
import random

import jsonlines
import numpy as np
import pandas as pd

papers = []
with jsonlines.open("/home/miao4/punim0521/NeuralAbstractiveSummarization/peersum/crawling_data/data/peersum_all.json",
                    "r") as reader:
    for paper in reader:
        papers.append(paper)

clusters = []
variances = []
variance_zero = 0
for paper in papers:
    summary = paper["meta_review"]
    paper_id = paper['paper_id']
    paper_acceptance = paper["paper_acceptance"]
    source_documents = []

    review_scores = []
    for i, review_i in enumerate(paper["reviews"]):
        if review_i["rating"] > 0:
            review_scores.append(review_i["rating"])
    variance = np.var(review_scores)
    variances.append(variance)

    if variance == 0:
        variance_zero += 1

    source_documents.append(paper["paper_abstract"])
    for review in paper["reviews"]:
        if "comment" in review.keys():
            text = review["comment"]
            source_documents.append(text)

    if summary.strip() != "" and len(source_documents) > 1:
        clusters.append(
            {"source_documents": source_documents, "review_score_variance": variance, "summary": summary,
             "paper_id": paper_id, "label": paper["label"], "paper_acceptance": paper_acceptance})
    # else:
    #     print(paper["paper_id"])

print("Max variance", np.max(variances))
print("Min variance", np.min(variances))
bins = [i * 1 for i in range(13)]
cats = pd.cut(variances, bins, right=False, include_lowest=True)
print(pd.value_counts(cats, sort=False))
print("papers with variance zero", variance_zero)

variance_low = []
variance_medium = []
variance_high = []
for cluster in clusters:
    review_score_variance = cluster["review_score_variance"]
    if 0 <= review_score_variance < 2:
        variance_low.append(cluster)
    if 2 <= review_score_variance < 8:
        variance_medium.append(cluster)
    if 8 <= review_score_variance < 12:
        variance_high.append(cluster)

final_clusters = []
final_clusters.extend(random.sample(variance_low, 10))
final_clusters.extend(random.sample(variance_medium, 10))
final_clusters.extend(random.sample(variance_high, 10))
random.shuffle(final_clusters)

with jsonlines.open("variance_high.json", "w") as writer:
    writer.write_all(variance_high)
