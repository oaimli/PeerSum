# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec


class EpochCallback(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print("Epoch", self.epoch, "Loss", loss)
        self.epoch += 1


if __name__ == "__main__":
    samples_tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(dm=0, dbow_words=0, vector_size=300, min_count=1, workers=multiprocessing.cpu_count() - 1,
                    compute_loss=True, epochs=100, window=15, sample=1e-5, callbacks=[EpochCallback()])
    model.build_vocab(samples_tagged)
    model.train(samples_tagged, total_examples=model.corpus_count, epochs=model.epochs, start_alpha=0.025, compute_loss=True, callbacks=[EpochCallback()])
    print("Loss", model.running_training_loss)
    print(model.infer_vector(["system", "response"]))