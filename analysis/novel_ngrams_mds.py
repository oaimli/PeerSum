# analysis for different mds preparing_data
import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import functools
import multiprocessing
import random

sys.path.append("../../")
from peersum.preparing_data.peersum_loader import loading_peersum
from preparing_data.data_loader import loading_mds

def novel_ngrams_multi_process(i, source_documents, summaries, analyzer):
    c = source_documents[i]
    s = summaries[i]

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


def novel_ngrams(source_documents, summaries, ngrams=1):
    documents_all = []
    for c, s in zip(source_documents, summaries):
        documents_all.append(s)
        documents_all.extend(c)
    count_vect = CountVectorizer(ngram_range=(ngrams, ngrams), min_df=1)
    count_vect.fit(documents_all)
    analyzer = count_vect.build_analyzer()

    count_all = len(summaries)
    partial_novel_grams = functools.partial(novel_ngrams_multi_process, source_documents=source_documents, summaries=summaries, analyzer=analyzer)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_novel_ngram_scores = list(p.imap(partial_novel_grams, range(count_all), chunksize=128))
    print("Novel %d-grams" % ngrams, np.mean(all_novel_ngram_scores))


if __name__ == "__main__":

    print("PeerSum only using official reviews")
    source_documents, summaries = loading_peersum(
        "../../peersum", data_name="peersum_lemma_stop", including_public_comments=False, including_responses=False, spliting=False)
    for n in [3, 4]:
        novel_ngrams(source_documents, summaries, ngrams=n)

    print("PeerSum including public comments")
    source_documents, summaries = loading_peersum(
        "../../peersum", data_name="peersum_lemma_stop", including_public_comments=True, including_responses=False,
        spliting=False)
    for n in [1, 2, 3, 4]:
        novel_ngrams(source_documents, summaries, ngrams=n)

    print("PeerSum including public comments and responses")
    source_documents, summaries = loading_peersum(
        "../../peersum", including_public_comments=True, including_responses=True, data_name="peersum_lemma_stop", spliting=False)
    for n in [1, 2, 3, 4]:
        novel_ngrams(source_documents, summaries, ngrams=n)



    # print("Multi-News")
    # source_documents, summaries = loading_mds(
    #     "../../preparing_data/", "multinews_lemma_stop", spliting=False)
    # count_all = len(summaries)
    # indexes = range(count_all)
    # target_indexes = random.sample(indexes, 5000)
    # source_documents_tmp = []
    # summaries_tmp = []
    # for i in target_indexes:
    #     source_documents_tmp.append(source_documents[i])
    #     summaries_tmp.append(summaries[i])
    # for n in [1, 2, 3, 4]:
    #     novel_ngrams(source_documents_tmp, summaries_tmp, ngrams=n)
    #
    #
    #
    # print("WCEP")
    # source_documents, summaries = loading_mds(
    #     "../../preparing_data/", "wcep_lemma_stop", spliting=False)
    # count_all = len(summaries)
    # indexes = range(count_all)
    # target_indexes = random.sample(indexes, 2000)
    # source_documents_tmp = []
    # summaries_tmp = []
    # for i in target_indexes:
    #     source_documents_tmp.append(source_documents[i])
    #     summaries_tmp.append(summaries[i])
    # for n in [1, 2, 3, 4]:
    #     novel_ngrams(source_documents_tmp, summaries_tmp, ngrams=n)
    #
    #
    #
    # print("Multi-XScience")
    # source_documents, summaries = loading_mds(
    #     "../../preparing_data/", "multixscience_lemma_stop", spliting=False)
    # count_all = len(summaries)
    # indexes = range(count_all)
    # target_indexes = random.sample(indexes, 5000)
    # source_documents_tmp = []
    # summaries_tmp = []
    # for i in target_indexes:
    #     source_documents_tmp.append(source_documents[i])
    #     summaries_tmp.append(summaries[i])
    # for n in [1, 2, 3, 4]:
    #     novel_ngrams(source_documents_tmp, summaries_tmp, ngrams=n)



    print("Wikisum")
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "wikisum_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 10000)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    for n in [2, 3, 4]:
        novel_ngrams(source_documents_tmp, summaries_tmp, ngrams=n)