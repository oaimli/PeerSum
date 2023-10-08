# analysis for different mds preparing_data
import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import functools
import multiprocessing
import random

sys.path.append("../../")
from peersum.loading_data.peersum_loader import loading_peersum
from peersum.loading_data.mds_loader import loading_mds
from utils.cleaning import cleaning_document, cleaning_documents


def novel_ngrams_multi_process(i, samples, analyzer):
    sample = samples[i]
    c = cleaning_documents(sample["source_documents"], multi_process=False, lemmatization=True,
                           removing_stop_words=True, sentence_split=False)
    s = cleaning_document(sample["summary"], lemmatization=True,
                          removing_stop_words=True, sentence_split=False)

    ngrams_list_summary = analyzer(s)
    ngrams_count_summary = len(ngrams_list_summary)

    ngrams_set_source_documents = []
    for d in c:
        ngrams_set_d = analyzer(d)
        ngrams_set_source_documents.extend(ngrams_set_d)

    # % novel n-grams
    diff = []
    for w in ngrams_list_summary:
        if w not in ngrams_set_source_documents:
            diff.append(w)
    if ngrams_count_summary == 0:
        score = 0
    else:
        score = len(diff) / ngrams_count_summary
    return score


def novel_ngrams(samples, ngrams=1):
    documents_all = []
    for item in samples:
        documents_all.append(item["summary"])
        documents_all.extend(item["source_documents"])
    count_vect = CountVectorizer(ngram_range=(ngrams, ngrams), min_df=1)
    count_vect.fit(documents_all)
    analyzer = count_vect.build_analyzer()

    count_all = len(samples)
    partial_novel_grams = functools.partial(novel_ngrams_multi_process, samples=samples, analyzer=analyzer)
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        all_novel_ngram_scores = list(p.imap(partial_novel_grams, range(count_all), chunksize=128))
    print("Novel %d-grams" % ngrams, np.mean(all_novel_ngram_scores))


if __name__ == "__main__":
    sampling_count = 1024
    print("PeerSum")
    papers = loading_peersum(including_public=True, including_author=True, including_abstract=True)
    papers = random.sample(papers, sampling_count)
    for n in [1, 2, 3]:
        novel_ngrams(papers, ngrams=n)

    # print("Multi-News")
    # samples = loading_mds(folder="../../datasets", data_name="multinews")
    # samples = random.sample(samples, sampling_count)
    # for n in [1, 2, 3]:
    #     novel_ngrams(samples, ngrams=n)
    # del samples
    #
    # print("WCEP")
    # samples = loading_mds(folder="../../datasets", data_name="wcep_100")
    # samples = random.sample(samples, sampling_count)
    # for n in [1, 2, 3]:
    #     novel_ngrams(samples, ngrams=n)
    # del samples
    #
    # print("Multi-XScience")
    # samples = loading_mds(folder="../../datasets", data_name="multixscience")
    # samples = random.sample(samples, sampling_count)
    # for n in [1, 2, 3]:
    #     novel_ngrams(samples, ngrams=n)
    # del samples
    #
    # print("Wikisum")
    # samples = loading_mds(folder="../../datasets", data_name="wikisum")
    # samples = random.sample(samples, sampling_count)
    # for n in [1, 2, 3]:
    #     novel_ngrams(samples, ngrams=n)
    # del samples

