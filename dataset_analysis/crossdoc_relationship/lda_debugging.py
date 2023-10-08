from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.callbacks import PerplexityMetric, Callback


lda = LdaModel(corpus=common_corpus, id2word=common_dictionary, num_topics=10, chunksize=128, iterations=50, passes=10)
for i in range(20000):
    lda.update(common_corpus)
    print(lda.log_perplexity(chunk=common_corpus))

# # model evaluation
# evals = []
# for i in range(20):
#     eval = lda.log_perplexity(chunk=corpus)
#     print("Pass", 10+i, eval)
#     lda.update(corpus)
#
# for i in range(len(common_texts)):
#     doc_i = common_texts[i]
#     topic_s = [0.0]*lda.num_topics
#     for item in lda[dictionary.doc2bow(doc_i)]:
#     lda.num_topics
#     print(topic_s)
#     for j in range(i+1, len(common_texts)):
#         doc_j = common_texts[j]
#         topic_d = lda[dictionary.doc2bow(doc_j)]
#         print(cosine_similarity([topic_s], [topic_d])[0][0])
