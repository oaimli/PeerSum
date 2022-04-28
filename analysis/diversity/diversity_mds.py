# diversity for different mds preparing_data
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functools
import multiprocessing
import random
import os

sys.path.append("../../../")
from preparing_data.data_loader import loading_mds
from peersum.preparing_data.peersum_loader import loading_peersum


def diversity_multi_process(i, source_documents, summaries):
    c = source_documents[i]
    s = summaries[i]

    word_list_summary = s.split()

    word_list_source_documents = []
    for d in c:
        word_list_d = d.split()
        word_list_source_documents.extend(word_list_d)

    # compression
    compression = len(word_list_source_documents) / len(word_list_summary)

    # coverage and density
    extractive_fragments = []
    m = 0
    n = 0
    while m < len(word_list_summary):
        f = []
        while n < len(word_list_source_documents):
            f_tmp = []
            if word_list_summary[m] == word_list_source_documents[n]:
                it = m
                jt = n
                while it < len(word_list_summary) and jt < len(word_list_source_documents) and \
                        word_list_summary[it] == word_list_source_documents[jt]:
                    f_tmp.append(word_list_summary[it])
                    it += 1
                    jt += 1
                if len(f) < len(f_tmp):
                    f = f_tmp
                n = jt
            else:
                n = n + 1
        m = m + np.max([len(f), 1])
        n = 0
        # print(f)
        extractive_fragments.append(f)
    # print(extractive_fragments)
    coverage = 0
    density = 0
    for fragment in extractive_fragments:
        coverage += len(fragment)
        density += len(fragment) * len(fragment)
    # print((coverage/len(word_set_meta_summary), density/len(word_set_meta_summary)))
    coverage = coverage / len(word_list_summary)
    density = density / len(word_list_summary)

    return {"compression": compression, "coverage": coverage, "density":density}



def diversity(source_documents, summaries, dataset_name):
    count_all = len(summaries)
    partial_diversity = functools.partial(diversity_multi_process, source_documents=source_documents, summaries=summaries)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        all_diversities = list(p.imap(partial_diversity, range(count_all), chunksize=128))

    all_compression = []
    all_coverage = []
    all_density = []
    for diversity in all_diversities:
        all_compression.append(diversity["compression"])
        all_coverage.append(diversity["coverage"])
        all_density.append(diversity["density"])

    print("Compression", np.median(all_compression))
    # plot coverage and density
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.kdeplot(x=all_coverage, y=all_density, shade=True, gridsize=50, clip=[(0, 1), (0, 10)], cmap="BuGn_d")
    fig_folder = "diversity_figs"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    plt.savefig(fig_folder + '/%s.jpg' % dataset_name)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    print("PeerSum only using official reviews")
    source_documents, summaries = loading_peersum(
        folder="../../../peersum", including_public_comments=False, including_responses=False, data_name="peersum_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1000)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "peersum_r")

    print("PeerSum including public comments")
    source_documents, summaries = loading_peersum(
        "../../../peersum", including_public_comments=True, including_responses=False, data_name="peersum_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1000)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "peersum_rc")

    print("PeerSum including public comments and responses")
    source_documents, summaries = loading_peersum(
        "../../../peersum", including_public_comments=True, including_responses=True, data_name="peersum_lemma_stop",
        spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1000)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "peersum_all")

    print("Multi-News")
    source_documents, summaries = loading_mds(
        "../../../preparing_data/", "multinews_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1500)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "multinews")

    print("WCEP")
    source_documents, summaries = loading_mds(
        "../../../preparing_data/", "wcep_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1000)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "wcep")

    print("Multi-XScience")
    source_documents, summaries = loading_mds(
        "../../../preparing_data/", "multixscience_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 1300)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "multixscience")

    print("Wikisum")
    source_documents, summaries = loading_mds(
        "../../../preparing_data/", "wikisum_lemma_stop", spliting=False)
    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, 3500)
    source_documents_tmp = []
    summaries_tmp = []
    for i in target_indexes:
        source_documents_tmp.append(source_documents[i])
        summaries_tmp.append(summaries[i])
    diversity(source_documents_tmp, summaries_tmp, "wikisum")