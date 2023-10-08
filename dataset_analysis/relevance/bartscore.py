import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import functools
import multiprocessing
import random

sys.path.append("../../../")
from peersum.loading_data.peersum_loader import loading_peersum
from peersum.loading_data.mds_loader import loading_mds
from utils.metrics import bart_score

def relevance_score(samples, scorer):
    avgs = []
    variances = []
    for sample in samples:
        source_documents = sample["source_documents"]
        summary = sample["summary"]
        candidates = []
        references = []
        for source_document in source_documents:
            candidates.append(source_document)
            references.append(summary)
        bart_scores = bart_score(candidates, references, bart_scorer=scorer)
        avgs.append(np.mean(bart_scores))
        variances.append(np.var(bart_scores))

    return {"mean": np.mean(avgs), "variance": np.mean(variances)}


if __name__ == "__main__":
    from utils.BARTScore.bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

    sampling_count = 1024
    print("PeerSum")
    samples = loading_peersum(folder="../../../peersum", including_public=True, including_author=True, including_abstract=True)
    samples = random.sample(samples, sampling_count)
    score_dict = relevance_score(samples, scorer=bart_scorer)
    print("mean", score_dict["mean"], "variance", score_dict["variance"])
    del samples

    # print("Multi-News")
    # samples = loading_mds(folder="../../../datasets", data_name="multinews")
    # samples = random.sample(samples, sampling_count)
    # score_dict = relevance_score(samples, scorer=bart_scorer)
    # print("mean", score_dict["mean"], "variance", score_dict["variance"])
    # del samples
    #
    # print("WCEP")
    # samples = loading_mds(folder="../../../datasets", data_name="wcep_100")
    # samples = random.sample(samples, sampling_count)
    # score_dict = relevance_score(samples, scorer=bart_scorer)
    # print("mean", score_dict["mean"], "variance", score_dict["variance"])
    # del samples
    #
    # print("Multi-XScience")
    # samples = loading_mds(folder="../../../datasets", data_name="multixscience")
    # samples = random.sample(samples, sampling_count)
    # score_dict = relevance_score(samples, scorer=bart_scorer)
    # print("mean", score_dict["mean"], "variance", score_dict["variance"])
    # del samples
    #
    # print("Wikisum")
    # samples = loading_mds(folder="../../../datasets", data_name="wikisum")
    # samples = random.sample(samples, sampling_count)
    # score_dict = relevance_score(samples, scorer=bart_scorer)
    # print("mean", score_dict["mean"], "variance", score_dict["variance"])
    # del samples