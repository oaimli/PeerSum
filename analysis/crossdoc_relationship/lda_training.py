from gensim import corpora, models
import sys
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import functools
import time

sys.path.append("../../../")
from peersum.preparing_data.peersum_loader import loading_peersum
from preparing_data.data_loader import loading_mds

def transform_vector(topic_distribution, num_topics):
    topic_vector = [0.0]*num_topics
    for item in topic_distribution:
        topic_vector[item[0]] = item[1]
    return topic_vector


def lda_prediction_multi_process(i, target_indexes, source_documents, summaries, lda_model, dictionary):
    cluster_index = target_indexes[i]
    c = source_documents[cluster_index]
    s = summaries[cluster_index]
    topic_s = lda_model[dictionary.doc2bow(s.split())]
    similarities = []
    for d in c:
        topic_d = lda_model[dictionary.doc2bow(d.split())]
        similarities.append(cosine_similarity([transform_vector(topic_s, lda_model.num_topics)], [transform_vector(topic_d, lda_model.num_topics)])[0][0])
    return {"variance":np.var(similarities), "mean":np.mean(similarities)}


def train_lda(source_documents, summaries, model_name, num_topics, iterations, min_passes, max_passes, prediction_sampling):
    print("train", model_name)
    all_documents = []
    for ds in source_documents:
        for d in ds:
            words = d.split()
            all_documents.append(words)
    for s in summaries:
        words = s.split()
        all_documents.append(words)

    dictionary = corpora.Dictionary(all_documents)

    corpus = [dictionary.doc2bow(words) for words in all_documents]
    del all_documents
    lda = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=multiprocessing.cpu_count()-1, chunksize=128, iterations=iterations, passes=min_passes)

    # model evaluation
    for i in range(max_passes-min_passes):
        eval = lda.log_perplexity(chunk=corpus)
        print("Pass", min_passes+i, eval)
        lda.update(corpus)
    # lda.save(model_name)

    count_all = len(summaries)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, int(count_all * prediction_sampling))

    partial_lda_prediction = functools.partial(lda_prediction_multi_process, target_indexes=target_indexes,
                                             source_documents=source_documents, summaries=summaries, lda_model=lda, dictionary=dictionary)
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        results = list(p.imap(partial_lda_prediction, range(len(target_indexes)), chunksize=128))

    lda_similarity_variances = []
    lda_similarity_means = []
    for item in results:
        lda_similarity_variances.append(item["variance"])
        lda_similarity_means.append(item["mean"])
    print("LDA similarity variance", np.mean(lda_similarity_variances), "mean", np.mean(lda_similarity_means))



if __name__=="__main__":
    print("Peersum raw pre_processing for debugging")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_peersum(
        "../../peersum", including_public_comments=True, data_name="peersum_lemma_stop", spliting=False)
    model_name = "lda_models/lda_peersum_debugging.model"
    train_lda(source_documents[:20], summaries[:20], model_name, num_topics=50, iterations=50, min_passes=2, max_passes=3, prediction_sampling=0.1)
    del source_documents, summaries

    # PeerSum
    print("Peersum with public comments")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_peersum(
        "../../peersum", including_public_comments=True, data_name="peersum_lemma_stop", spliting=False)
    model_name = "lda_models/lda_peersum.model"
    train_lda(source_documents, summaries, model_name, num_topics=50, iterations=50, min_passes=100,
              max_passes=800, prediction_sampling=0.8)
    del source_documents, summaries


    # WCEP
    print("WCEP")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "wcep_lemma_stop", spliting=False)
    model_name = "lda_models/lda_wcep.model"
    train_lda(source_documents, summaries, model_name, num_topics=50, iterations=50, min_passes=100,
              max_passes=800, prediction_sampling=0.8)
    del source_documents, summaries


    # Multi-XScience
    print("Mutli-XScience")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "multixscience_lemma_stop", spliting=False)
    model_name = "lda_models/lda_multixscience.model"
    train_lda(source_documents, summaries, model_name, num_topics=50, iterations=50, min_passes=100,
              max_passes=600, prediction_sampling=0.8)
    del source_documents, summaries


    # Multi-News
    print("Multi-News")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "multinews_lemma_stop", spliting=False)
    model_name = "lda_models/lda_multinews.model"
    train_lda(source_documents, summaries, model_name, num_topics=50, iterations=50, min_passes=100,
              max_passes=600, prediction_sampling=0.8)
    del source_documents, summaries

    # WikiSum
    print("WikiSum")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "wikisum_lemma_stop", spliting=False)
    model_name = "lda_models/lda_wikisum.model"
    train_lda(source_documents, summaries, model_name, num_topics=50, iterations=50, min_passes=80,
              max_passes=400, prediction_sampling=0.5)
    del source_documents, summaries