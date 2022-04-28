# basic statistics for different mds preparing_data
import sys
import numpy as np
import random

sys.path.append("../../")
from preparing_data.data_loader import loading_mds
import functools
import multiprocessing
from peersum.preparing_data.peersum_loader import loading_peersum

def statistics_multi_process(i, source_documents, summaries):
    vocab_summary = set([])
    vocab_document = set([])

    s = summaries[i]
    c = source_documents[i]
    sents_summary = s.split("sentence_split")
    token_list_summary = " ".join(sents_summary).split()
    vocab_summary.update(set(token_list_summary))

    count_source_documents = len(c)

    cluster_sent_count_source_document = []
    cluster_token_count_source_document = []

    for d in c:
        sents_d = d.split("sentence_split")
        cluster_sent_count_source_document.append(len(sents_d))
        token_list_d = " ".join(sents_d).split()
        # print(token_list_d)
        vocab_document.update(set(token_list_d))
        cluster_token_count_source_document.append(len(token_list_d))

    return {"sent_count_summary":len(sents_summary), "token_count_summary":len(token_list_summary), "count_source_documents":count_source_documents, "sent_count_source_document":np.mean(cluster_sent_count_source_document), "token_count_source_document":np.mean(cluster_token_count_source_document), "vocab_summary":vocab_summary, "vocab_document":vocab_document}


def statistics(source_documents, summaries):
    count_all = len(summaries)
    partial_statistics = functools.partial(statistics_multi_process,
                                                    source_documents=source_documents,
                                                    summaries=summaries)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        results = list(p.imap(partial_statistics, range(count_all), chunksize=128))
    print("Results", len(results))


    all_token_count_summary = []
    all_sent_count_summary = []
    all_count_source_documents = []
    all_token_count_source_document = []
    all_sent_count_source_document = []

    source_document_count_dict = {}  # The number of papers that have specific number of source documents
    vocab_summary = set([])
    vocab_document = set([])

    for item in results:
        all_sent_count_summary.append(item["sent_count_summary"])
        all_token_count_summary.append(item["token_count_summary"])
        all_count_source_documents.append(item["count_source_documents"])
        all_token_count_source_document.append(item["token_count_source_document"])
        all_sent_count_source_document.append(item["sent_count_source_document"])
        source_document_count_dict[item["count_source_documents"]] = source_document_count_dict.get(item["count_source_documents"], 0) + 1
        vocab_summary.update(item["vocab_summary"])
        vocab_document.update(item["vocab_document"])


    print("avg_token_count_summary", np.mean(all_token_count_summary))
    print("avg_sent_count_summary", np.mean(all_sent_count_summary))
    print("avg_count_source_documents", np.mean(all_count_source_documents))
    print("avg_token_count_source_document", np.mean(all_token_count_source_document))
    print("avg_sent_count_source_document", np.mean(all_sent_count_source_document))
    print("source documents count dict", source_document_count_dict)
    print("vocabunary summary", len(vocab_summary))
    print("vocabunary_document", len(vocab_document))


if __name__ == "__main__":
    print("PeerSum only official reviews")
    source_documents, summaries = loading_peersum(
        "../../peersum", including_public_comments=False, including_responses=False, data_name="peersum_cleaned",
        spliting=False)
    statistics(source_documents, summaries)
    del source_documents, summaries

    print("PeerSum including public comments")
    source_documents, summaries = loading_peersum("../../peersum", including_public_comments=True, including_responses=False, data_name="peersum_cleaned", spliting=False)
    statistics(source_documents, summaries)
    del source_documents, summaries

    print("PeerSum including public comments and responses")
    source_documents, summaries = loading_peersum(
        "../../peersum", including_public_comments=True, including_responses=True, data_name="peersum_cleaned", spliting=False)
    statistics(source_documents, summaries)
    del source_documents, summaries


    # print("Multi-News")
    # source_documents, summaries = loading_mds("../../preparing_data/", "multinews_lemma_stop", spliting=False)
    # statistics(source_documents, summaries)
    # del source_documents, summaries
    #
    #
    # print("WCEP")
    # source_documents, summaries = loading_mds("../../preparing_data/", "wcep_lemma_stop", spliting=False)
    # statistics(source_documents, summaries)
    # del source_documents, summaries
    #
    #
    # print("Multi-XScience")
    # source_documents, summaries = loading_mds("../../preparing_data/", "multixscience_lemma_stop", spliting=False)
    # statistics(source_documents, summaries)
    # del source_documents, summaries
    #
    #
    # print("Wikisum lemma stop")
    # source_documents, summaries = loading_mds(
    #     "../../preparing_data/", "wikisum_lemma_stop", spliting=False)
    # count_all = len(summaries)
    # indexes = range(count_all)
    # target_indexes = random.sample(indexes, 50000)
    # source_documents_tmp = []
    # summaries_tmp = []
    # for i in target_indexes:
    #     source_documents_tmp.append(source_documents[i])
    #     summaries_tmp.append(summaries[i])
    # statistics(source_documents_tmp, summaries_tmp)
    # del source_documents, summaries



