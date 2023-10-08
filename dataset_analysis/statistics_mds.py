# basic statistics for different mds preparing_data
import sys
import numpy as np
import random
from nltk.tokenize import sent_tokenize
import functools
import multiprocessing

sys.path.append("../../")
from loading_data.mds_loader import loading_mds
from loading_data.peersum_loader import loading_peersum
from utils.cleaning import cleaning_document, cleaning_documents


def statistics_multi_process(i, samples):
    vocab_summary = set([])
    vocab_document = set([])

    sample = samples[i]
    s = cleaning_document(sample["summary"], lemmatization=False, removing_stop_words=False, sentence_split=False)
    c = cleaning_documents(sample["source_documents"], multi_process=False, lemmatization=False,
                           removing_stop_words=False, sentence_split=False)
    sents_summary = sent_tokenize(s)
    token_list_summary = " ".join(sents_summary).split()
    vocab_summary.update(set(token_list_summary))

    count_source_documents = len(c)

    cluster_sent_count_source_document = []
    cluster_token_count_source_document = []

    for d in c:
        sents_d = sent_tokenize(d)
        cluster_sent_count_source_document.append(len(sents_d))
        token_list_d = " ".join(sents_d).split()
        # print(token_list_d)
        vocab_document.update(set(token_list_d))
        cluster_token_count_source_document.append(len(token_list_d))

    return {"sent_count_summary": len(sents_summary), "token_count_summary": len(token_list_summary),
            "count_source_documents": count_source_documents,
            "sent_count_source_document": np.mean(cluster_sent_count_source_document) if len(cluster_sent_count_source_document)>0 else 0,
            "token_count_source_document": np.mean(cluster_token_count_source_document) if len(cluster_token_count_source_document)>0 else 0, "vocab_summary": vocab_summary,
            "vocab_document": vocab_document}


def statistics(samples):
    count_all = len(samples)
    partial_statistics = functools.partial(statistics_multi_process, samples=samples)
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
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
        source_document_count_dict[item["count_source_documents"]] = source_document_count_dict.get(
            item["count_source_documents"], 0) + 1
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
    sampling_count = 1024
    print("PeerSum")
    papers = loading_peersum(including_public=True, including_author=True, including_abstract=True)
    papers = random.sample(papers, sampling_count)
    statistics(papers)
    #
    # print("Multi-News")
    # samples = loading_mds(folder="../../datasets", data_name="multinews")
    # samples = random.sample(samples, sampling_count)
    # statistics(samples)
    # del samples
    #
    # print("WCEP")
    # samples = loading_mds(folder="../../datasets", data_name="wcep_100")
    # samples = random.sample(samples, sampling_count)
    # statistics(samples)
    # del samples
    #
    # print("Multi-XScience")
    # samples = loading_mds(folder="../../datasets", data_name="multixscience")
    # samples = random.sample(samples, sampling_count)
    # statistics(samples)
    # del samples
    #
    # print("Wikisum")
    # samples = loading_mds(folder="../../datasets", data_name="wikisum")
    # samples = random.sample(samples, sampling_count)
    # statistics(samples)
    # del samples
    #
    # print("arxiv")
    # samples = loading_mds(folder="../../datasets", data_name="arxiv")
    # print("arxiv", len(samples))
    # samples = random.sample(samples, sampling_count)
    # statistics(samples)
    # del samples
