import os.path
import numpy as np
import jsonlines
from nltk.tokenize import sent_tokenize

folder = "annotated_results"
samples = []
for i in range(10):
    print(i)
    with jsonlines.open(os.path.join(folder, "peersum_sampled_%d.json" % i)) as reader:
        for line in reader:
            samples.append(line)


variance_low_words = []
variance_medium_words = []
variance_high_words = []
variance_low_sents = []
variance_medium_sents = []
variance_high_sents = []
for sample in samples:
    anchored_text = sample["anchored_texts"]
    summary = sample["summary"]
    summary_sents = sent_tokenize(summary)
    anchored_sents = []
    for sent in summary_sents:
        if sent in anchored_text:
            anchored_sents.append(sent)
    review_score_variance = sample["review_score_variance"]
    ratio_words = len(anchored_text.split()) / len(summary.split())
    ratio_sents = len(anchored_sents) / len(summary_sents)
    if 0 <= review_score_variance < 2:
        variance_low_words.append(ratio_words)
        variance_low_sents.append(ratio_sents)
    if 2 <= review_score_variance < 8:
        variance_medium_words.append(ratio_words)
        variance_medium_sents.append(ratio_sents)
    if 8 <= review_score_variance < 12:
        variance_high_words.append(ratio_words)
        variance_high_sents.append(ratio_sents)

print("low", np.mean(variance_low_words))
print("medium", np.mean(variance_medium_words))
print("high", np.mean(variance_high_words))

print("low", np.mean(variance_low_sents))
print("medium", np.mean(variance_medium_sents))
print("high", np.mean(variance_high_sents))


all_papers = []
with jsonlines.open("/home/miao4/punim0521/NeuralAbstractiveSummarization/peersum/crawling_data/data/peersum_all.json",
                    "r") as reader:
    for paper in reader:
        all_papers.append(paper)

clusters = {}
for paper in all_papers:
    summary = paper["meta_review"]
    paper_id = paper['paper_id']
    paper_acceptance = paper["paper_acceptance"]
    source_documents = []

    review_scores = []
    for i, review_i in enumerate(paper["reviews"]):
        if review_i["rating"] > 0:
            review_scores.append(review_i["rating"])

    source_documents.append(paper["paper_abstract"])
    for review in paper["reviews"]:
        if "comment" in review.keys():
            text = review["comment"]
            source_documents.append(text)

    if summary.strip() != "" and len(source_documents) > 1:
        clusters[summary] = review_scores


words_ratios_range_low = []
words_ratio_range_high = []
variances_range_low = []
variance_range_high = []
for sample in samples:
    anchored_text = sample["anchored_texts"]
    summary = sample["summary"]
    summary_sents = sent_tokenize(summary)
    anchored_sents = []
    for sent in summary_sents:
        if sent in anchored_text:
            anchored_sents.append(sent)
    review_scores = clusters[summary]
    review_score_variance = sample["review_score_variance"]
    ratio_words = len(anchored_text.split()) / len(summary.split())
    # ratio_sents = len(anchored_sents) / len(summary_sents)
    if np.max(review_scores)-np.min(review_scores) < 4:
        variances_range_low.append(review_score_variance)
        words_ratios_range_low.append(ratio_words)
    else:
        variance_range_high.append(review_score_variance)
        words_ratio_range_high.append(ratio_words)

print(variances_range_low)
print(variance_range_high)
print("low range", len(variances_range_low), np.mean(variances_range_low), np.mean(words_ratios_range_low))
print("high range", len(variance_range_high), np.mean(variance_range_high), np.mean(words_ratio_range_high))