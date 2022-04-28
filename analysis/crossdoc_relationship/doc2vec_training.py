# -*- coding: utf-8 -*-
import sys
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import time
import random
from gensim.models.callbacks import CallbackAny2Vec
import functools


sys.path.append("../../../")
from preparing_data.data_loader import loading_mds
from peersum.preparing_data.peersum_loader import loading_peersum

def prepare_samples_train(source_documents, summaries):
    samples_train = []
    cluster_index = 0
    for ds, s in zip(source_documents, summaries):
        words = s.split()
        # print("summary", len(words))
        samples_train.append(TaggedDocument(words, ["summary_%d"%cluster_index]))
        document_index = 0
        for d in ds:
            words = d.split()
            # print("document", len(words))
            samples_train.append(TaggedDocument(words, ["document_%d_%d"%(cluster_index, document_index)]))
            document_index += 1
        cluster_index += 1
    return samples_train


def get_variance_multi_process(i, target_indexes, source_documents, model):
    cluster_index = target_indexes[i]
    similarities = []
    document_index = 0
    for d in source_documents[cluster_index]:
        if d.strip()!="":
            similarities.append(model.docvecs.similarity("summary_%d"%cluster_index, "document_%d_%d"%(cluster_index, document_index)))
            document_index += 1
    return {"variance": np.var(similarities), "mean":np.mean(similarities)}

def get_variance(model_name, source_documents, sampling=0.1):
    model = Doc2Vec.load(model_name)

    count_all = len(source_documents)
    indexes = range(count_all)
    target_indexes = random.sample(indexes, int(count_all * sampling))

    partial_get_variance = functools.partial(get_variance_multi_process, target_indexes=target_indexes, source_documents=source_documents, model=model)
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
        results = list(p.imap(partial_get_variance, range(len(target_indexes)), chunksize=128))

    # only compute some pre_processing in the prediction
    variances = []
    means = []
    for item in results:
        variances.append(item["variance"])
        means.append(item["mean"])
    return np.mean(variances), np.mean(means)


class EpochCallback(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, model_name):
        self.epoch = 0
        # self.losses = []
        # items = model_name.split(".")
        # self.model_name = items[0]

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        # print("Epoch", self.epoch, "Loss", loss)
        # self.losses.append(loss)
        # if len(self.losses)>5 and np.var(self.losses[-6:-1])<0.001:
        #     model.save(self.model_name + "_%d"%self.epoch + ".model")
        self.epoch += 1


if __name__ == "__main__":
    model_folder = "doc_embedding/"

    # PeerSum
    print("Peersum with public comments")
    model_name = model_folder + 'doc2vec_peersum.model'
    source_documents, summaries = loading_peersum("../../peersum", including_public_comments=True, data_name="peersum_lemma_stop", spliting=False)
    samples = prepare_samples_train(source_documents, summaries)
    del summaries

    model = Doc2Vec(dm=0, dbow_words=0, vector_size=300, min_count=1, workers=multiprocessing.cpu_count()-1, compute_loss=True, epochs=800, window=15, sample=1e-5)
    model.build_vocab(samples)
    model.train(samples, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, callbacks=[EpochCallback(model_name)])
    model.save(model_name)
    print(model_name)
    print("Doc2Vec variance and mean", get_variance(model_name, source_documents, sampling=0.9))
    del source_documents, samples


    # WCEP
    print("WCEP")
    model_name = model_folder + 'doc2vec_wcep.model'
    source_documents, summaries = loading_mds("../../preparing_data/", "wcep_lemma_stop", spliting=False)
    samples = prepare_samples_train(source_documents, summaries)
    del summaries

    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, min_count=1, workers=multiprocessing.cpu_count(), compute_loss=True, epochs=800, window=15, sample=1e-5)
    model.build_vocab(samples)
    model.train(samples, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, callbacks=[EpochCallback(model_name)])
    model.save(model_name)
    print(model_name)
    print("Doc2Vec variance and mean", get_variance(model_name, source_documents, sampling=0.9))
    del source_documents, samples


    # Multi-XScience
    print("Mutli-XScience")
    model_name = model_folder + 'doc2vec_multixscience.model'
    source_documents, summaries = loading_mds("../../preparing_data/", "multixscience_lemma_stop", spliting=False)
    samples = prepare_samples_train(source_documents, summaries)
    del summaries

    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, min_count=1, workers=multiprocessing.cpu_count(), compute_loss=True, epochs=600, window=15, sample=1e-5)
    model.build_vocab(samples)
    model.train(samples, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, callbacks=[EpochCallback(model_name)])
    model.save(model_name)
    print(model_name)
    print("Doc2Vec variance and mean", get_variance(model_name, source_documents, sampling=0.8))
    del source_documents, samples


    # Multi-News
    print("Multi-News")
    model_name = model_folder + 'doc2vec_multinews.model'
    print(model_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds(
        "../../preparing_data/", "multinews_lemma_stop", spliting=False)
    samples = prepare_samples_train(source_documents, summaries)
    del summaries

    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, min_count=1, workers=multiprocessing.cpu_count(), compute_loss=True, epochs=600, window=15, sample=1e-5)
    model.build_vocab(samples)
    model.train(samples, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, callbacks=[EpochCallback(model_name)])
    model.save(model_name)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Doc2Vec variance and mean", get_variance(model_name, source_documents, sampling=0.8))
    del source_documents, samples


    # WikiSum
    print("WikiSum")
    model_name = model_folder + 'doc2vec_wikisum.model'
    print(model_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    source_documents, summaries = loading_mds("../../preparing_data/", "wikisum_lemma_stop", spliting=False)
    samples = prepare_samples_train(source_documents, summaries)
    del summaries

    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, min_count=1, workers=multiprocessing.cpu_count(), compute_loss=True, epochs=400, window=15, sample=1e-5)
    model.build_vocab(samples)
    model.train(samples, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, callbacks=[EpochCallback(model_name)])
    model.save(model_name)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Doc2Vec variance and mean", get_variance(model_name, source_documents, sampling=0.6))
    del source_documents, samples

