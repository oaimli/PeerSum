# analysis for different mds preparing_data
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader
import multiprocessing
import functools
import re
from sentence_transformers import SentenceTransformer, util

sys.path.append("../../../")
from peersum.loading_data.mds_loader import loading_mds
from peersum.loading_data.peersum_loader import loading_peersum
from utils.metrics import rouge, bert_score


def gensim_glove(model="glove-wiki-gigaword-300"):
    glove_vectors = gensim.downloader.load(model)
    return glove_vectors


def rouge_variance_multi_process(i, source_documents, summaries):
    c = source_documents[i]
    s = summaries[i]
    rouge1s_tmp = []
    rouge2s_tmp = []
    rougels_tmp = []
    for d in c:
        if len(d.strip())>2 and len(s.strip())>2:
            # print(len(d), len(s))
            d.replace("sentence_split", "\n")
            s.replace("sentence_split", "\n")
            scores = rouge(s, d, types=['rouge1', 'rouge2', 'rougeLsum'])
            rouge1s_tmp.append(scores["rouge1"]["r"])
            rouge2s_tmp.append(scores["rouge2"]["r"])
            rougels_tmp.append(scores["rougeLsum"]["f"])
    return {"rouge1_var":np.var(rouge1s_tmp), "rouge2_var":np.var(rouge2s_tmp), "rougel_var":np.var(rougels_tmp), "rouge1_mean":np.mean(rouge1s_tmp), "rouge2_mean":np.mean(rouge2s_tmp), "rougel_mean":np.mean(rougels_tmp)}

def rouge_variance(source_documents, summaries):

    count_all = len(summaries)
    partial_rouge_variance = functools.partial(rouge_variance_multi_process, source_documents=source_documents,
                                            summaries=summaries)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_rouges = list(p.imap(partial_rouge_variance, range(count_all), chunksize=128))

    rouge1s_var = []
    rouge2s_var = []
    rougels_var = []
    rouge1s_mean = []
    rouge2s_mean = []
    rougels_mean = []
    for item in all_rouges:
        rouge1s_var.append(item["rouge1_var"])
        rouge2s_var.append(item["rouge2_var"])
        rougels_var.append(item["rougel_var"])
        rouge1s_mean.append(item["rouge1_mean"])
        rouge2s_mean.append(item["rouge2_mean"])
        rougels_mean.append(item["rougel_mean"])
    print("Rouge 1 Recall variance", np.mean(rouge1s_var), "Rouge 1 Recall mean", np.mean(rouge1s_mean))
    print("Rouge 2 Recall variance", np.mean(rouge2s_var), "Rouge 2 Recall mean", np.mean(rouge2s_mean))
    print("Rouge l F1 variance", np.mean(rougels_var), "Rouge l F1 mean", np.mean(rougels_mean))


def novel_ngrams_variance_multi_process(i, source_documents, summaries, analyzer):
    c = source_documents[i]
    s = summaries[i]
    ngrams_list_summary = analyzer(s)
    ngrams_count_summary = len(ngrams_list_summary)

    scores_cluster = []
    for d in c:
        ngrams_set_source_document = analyzer(d)

        # % novel n-grams
        diff = []
        for w in ngrams_list_summary:
            if w not in ngrams_set_source_document:
                diff.append(w)
        if ngrams_count_summary == 0:
            score = 0
        else:
            score = len(diff) / ngrams_count_summary
        scores_cluster.append(score)
    return {"var":np.var(scores_cluster), "mean":np.mean(scores_cluster)}


def novel_ngrams_variance(source_documents, summaries, ngrams=1):
    documents_all = []
    for c, s in zip(source_documents, summaries):
        documents_all.append(s)
        for d in c:
            documents_all.append(d)
    count_vect = CountVectorizer(ngram_range=(ngrams, ngrams), min_df=1)
    count_vect.fit(documents_all)
    analyzer = count_vect.build_analyzer()

    count_all = len(summaries)
    partial_novel_ngrams_variance = functools.partial(novel_ngrams_variance_multi_process, source_documents=source_documents,
                                               summaries=summaries, analyzer=analyzer)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_scores = list(p.imap(partial_novel_ngrams_variance, range(count_all), chunksize=128))

    all_ngram_scores_var = []
    all_ngrams_scores_mean = []
    for item in all_scores:
        all_ngram_scores_var.append(item["var"])
        all_ngrams_scores_mean.append(item["mean"])
    print("Novel %d-grams" % ngrams, "variance", np.mean(all_ngram_scores_var), "mean", np.mean(all_ngrams_scores_mean))


def doc2vec_variance():
    print("Doc2Vec variance: please refer to doc2vec training")


def lda_variance():
    print("LDA variance: please refer to lda training")


def tfidf_weighted_embedding_variance_multi_process(i, source_documents, summaries, tfidf_vectorizer, word_embedding_matrix, word_embedding_matrix_len):
    c = source_documents[i]
    s = summaries[i]
    tfidf_summary = tfidf_vectorizer.transform([s])
    embedding_summary = np.dot(tfidf_summary[0].todense(), word_embedding_matrix) / word_embedding_matrix_len

    similatities_cluster_tfidf = []
    for d in c:
        tfidf_document = tfidf_vectorizer.transform([d])
        embedding_document = np.dot(tfidf_document[0].todense(), word_embedding_matrix) / word_embedding_matrix_len
        similatities_cluster_tfidf.append(cosine_similarity(embedding_summary, embedding_document)[0][0])
    return {"var":np.var(similatities_cluster_tfidf), "mean":np.mean(similatities_cluster_tfidf)}

def tfidf_weighted_embedding_variance(source_documents, summaries, word2vec):
    documents_all = []
    for c, s in zip(source_documents, summaries):
        documents_all.append(s)
        documents_all.extend(c)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(documents_all)

    word_embedding_matrix = []
    for w in tfidf_vectorizer.get_feature_names():
        # np.random.randn(300)
        word_embedding_matrix.append(word2vec.get_vector(w, norm=True) if word2vec.has_index_for(w) else np.random.randn(300))
    word_embedding_matrix = np.array(word_embedding_matrix)
    word_embedding_matrix_len = len(word_embedding_matrix)
    print("word embedding matrix shape", word_embedding_matrix.shape)

    count_all = len(summaries)
    partial_tfidf_weighted_embedding_variance = functools.partial(tfidf_weighted_embedding_variance_multi_process,
                                                    source_documents=source_documents,
                                                    summaries=summaries, tfidf_vectorizer=tfidf_vectorizer, word_embedding_matrix=word_embedding_matrix, word_embedding_matrix_len=word_embedding_matrix_len)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        results = list(p.imap(partial_tfidf_weighted_embedding_variance, range(count_all), chunksize=128))

    similarity_variance_tfidf = []
    similarity_mean_tfidf = []
    for item in results:
        similarity_variance_tfidf.append(item["var"])
        similarity_mean_tfidf.append(item["mean"])

    # similarity_variance_tfidf = []
    # similarity_mean_tfidf = []
    # for c, s in zip(source_documents, summaries):
    #     tfidf_summary = tfidf_vectorizer.transform([s])
    #     print("tfidf shape", tfidf_summary[0].shape)
    #     embedding_summary = np.dot(tfidf_summary[0].todense(), word_embedding_matrix) / word_embedding_matrix_len
    #     print("summary shape", embedding_summary.shape)
    #
    #     similatities_cluster_tfidf = []
    #     for d in c:
    #         tfidf_document = tfidf_vectorizer.transform([d])
    #         embedding_document = np.dot(tfidf_document[0].todense(), word_embedding_matrix) / word_embedding_matrix_len
    #         print("document shape", embedding_document.shape)
    #         similatities_cluster_tfidf.append(cosine_similarity([embedding_summary], [embedding_document])[0][0])
    #     similarity_variance_tfidf.append(np.var(similatities_cluster_tfidf))
    #     similarity_mean_tfidf.append(np.mean(similatities_cluster_tfidf))

    print("TFIDF weighted similarity variance", np.mean(similarity_variance_tfidf), "TFIDF weighted similarity mean", np.mean(similarity_mean_tfidf))


def wmdistance_variance_multi_process(i, source_documents, summaries, word2vec):
    s = re.sub('[,.!?;():\s]', ' ', summaries[i])
    c = source_documents[i]
    s = s.split()
    distance_cluster = []
    for d in c:
        # d = re.sub('[,.!?;():\s]', ' ', d)
        d = d.split()
        if len(s)>1 and len(d)>1:
            distance = word2vec.wmdistance(s, d)
            distance_cluster.append(distance)
    return {"distance_var": np.var(distance_cluster), "distance_mean":np.mean(distance_cluster)}


def wmdistance_variance(source_documents, summaries, word2vec):
    count_all = len(summaries)
    partial_wmdistance_variance = functools.partial(wmdistance_variance_multi_process,
                                                      source_documents=source_documents,
                                                      summaries=summaries, word2vec=word2vec)
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
        all_distances = list(p.imap(partial_wmdistance_variance, range(count_all), chunksize=128))

    distance_variances = []
    distance_means = []
    for item in all_distances:
        distance_variances.append(item["distance_var"])
        distance_means.append(item["distance_mean"])
    print("WMDistance variance", np.mean(distance_variances), "WMDistance mean",
          np.mean(distance_means))


def bert_score_similarity(source_documents, summaries):
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    bert_variances = []
    bert_means = []
    for c, s in zip(source_documents, summaries):
        bert_similarities = []
        for d in c:
            bert_similarities.append(bert_score(d, s, scorer)["f"])
        bert_variances.append(np.var(bert_similarities))
        bert_means.append(np.mean(bert_similarities))
    print("BERTScore variance", np.mean(bert_variances), "BERTScore mean", np.mean(bert_means))


def sampling(source_documents, summaries, sampling=0.01):
    if sampling <= 0:
        return source_documents, summaries
    else:
        count_all =len(summaries)
        indexes = range(count_all)
        if sampling>1:
            target_indexes = random.sample(indexes, int(sampling))
        else:
            target_indexes = random.sample(indexes, int(count_all*sampling))
        source_documents_tmp = []
        summaries_tmp = []
        for i in target_indexes:
            source_documents_tmp.append(source_documents[i])
            summaries_tmp.append(summaries[i])
        return source_documents_tmp, summaries_tmp



if __name__ == "__main__":
    sampling_count = 500

    word2vec = gensim_glove("glove-wiki-gigaword-300")
    # sentence_bert = SentenceTransformer('all-mpnet-base-v2')

    # print("Multi-News")
    # # source_documents, summaries = loading_mds("../../../preparing_data/", "multinews_cleaned", spliting=False)
    # source_documents, summaries = loading_mds("/scratch/miao4/datasets_tmp/", "multinews_cleaned", spliting=False)
    # source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    # rouge_variance(source_documents, summaries)
    # wmdistance_variance(source_documents, summaries, word2vec)
    # bert_score(source_documents, summaries)
    #
    #
    # print("PeerSum only including official reviews")
    # source_documents, summaries = loading_peersum("../../../peersum", including_public_comments=False, including_responses=False, data_name="peersum_cleaned")
    # source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    # rouge_variance(source_documents, summaries)
    # wmdistance_variance(source_documents, summaries, word2vec)
    # bert_score(source_documents, summaries)
    #
    # print("PeerSum including public comments")
    # source_documents, summaries = loading_peersum("../../../peersum", including_public_comments=True, including_responses=False,
    #                                               data_name="peersum_cleaned")
    # source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    # rouge_variance(source_documents, summaries)
    # wmdistance_variance(source_documents, summaries, word2vec)
    # bert_score(source_documents, summaries)

    print("PeerSum including public comments and responses")
    source_documents, summaries = loading_peersum("../../../peersum", including_public_comments=True, including_responses=True,
                                                  data_name="peersum_cleaned")
    source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    # rouge_variance(source_documents, summaries)
    wmdistance_variance(source_documents, summaries, word2vec)
    bert_score(source_documents, summaries)


    print("WCEP")
    # source_documents, summaries = loading_mds("../../../preparing_data/", "wcep_cleaned", spliting=False)
    source_documents, summaries = loading_mds("/scratch/miao4/datasets_tmp/", "wcep_cleaned", spliting=False)
    source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    rouge_variance(source_documents, summaries)
    wmdistance_variance(source_documents, summaries, word2vec)
    bert_score_similarity(source_documents, summaries)


    print("Multi-XScience")
    # source_documents, summaries = loading_mds("../../../preparing_data/", "multixscience_cleaned", spliting=False)
    source_documents, summaries = loading_mds("/scratch/miao4/datasets_tmp/", "multixscience_cleaned", spliting=False)
    source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    rouge_variance(source_documents, summaries)
    wmdistance_variance(source_documents, summaries, word2vec)
    bert_score_similarity(source_documents, summaries)


    print("Wikisum")
    # source_documents, summaries = loading_mds("../../../preparing_data/", "wikisum_cleaned", spliting=False)
    source_documents, summaries = loading_mds("/scratch/miao4/datasets_tmp/", "wikisum_cleaned", spliting=False)
    source_documents, summaries = sampling(source_documents, summaries, sampling=sampling_count)
    rouge_variance(source_documents, summaries)
    wmdistance_variance(source_documents, summaries, word2vec)
    bert_score_similarity(source_documents, summaries)






