# import numpy as np
# word2vec = {}
# with open("/home/miao4/pre_processing/glove/glove.840B.300d.txt", "r") as f:
#     lines = f.readlines()
# print(lines[0], len(lines[0].split()))
# print(lines[1])
# print(np.asarray([float(i) for i in lines[-1].split()[1:]]))


# import numpy as np
# a = np.asarray([[1,2], [3, 4]])
# print(a.shape)
# b = np.asarray([[1, 2, 3], [3, 4, 5]])
# print(b.shape)
# c = np.dot(a[0], b)
# print(c.shape)
# print(c/2)
# d = np.asarray([1, 2])
# e = np.asarray([[2, 3], [4, 5]])
# f = np.dot(d, e)
# print(f)
# from sklearn.metrics.pairwise import cosine_similarity
# print(cosine_similarity([c, f], [f])[0][0])

# h = "Flood Warning issued March 16 at 9:32PM CDT expiring March 19 at 7:51PM CDT in effect for: Shelby Flood Warning issued March 16 at 9:32PM CDT expiring March 18 at 1:00PM CDT in effect for: Lake Flood Warning issued March 16 at 9:32PM CDT expiring March 20 at 7:00AM CDT in effect for: Dyer, Lake, Lauderdale Flood Warning issued March 16 at 9:32PM CDT expiring March 21 at 10:00AM CDT in effect for: Lauderdale, Tipton Flood Warning issued March 16 at 8:35PM CDT expiring March 17 at 8:12AM CDT in effect for: Livingston, McCracken"
# r = "For the first time in the history of the Division I men's tournament, a #16 seed defeats a #1 seed, as UMBC shocks top overall seed Virginia 74–54."
# from rouge import Rouge
# rouge = Rouge()
# scores = rouge.get_scores(h, r)[0]
# print(scores["rouge-1"]["r"])


import sys

sys.path.append("../../")
# from utils.text_cleaning import cleaning_document, initializing_spacy
#
#
# def processsing_peersum_multi_process(i, papers, nlp, lemmatization, removing_stop_words, sentence_split):
#     paper = papers[i]
#
#     paper["meta_review"] = cleaning_document(paper["meta_review"], nlp, lemmatization, removing_stop_words, sentence_split)
#
#     for review in paper["reviews"]:
#         review_text = review["content"]["comment"]
#         review["content"]["comment"] = cleaning_document(review_text, nlp, lemmatization, removing_stop_words, sentence_split)
#     return paper
#
#
# def processing_peersum(papers, lemmatization=True, removing_stop_words=True, sentence_split=True):
#     nlp = initializing_spacy()
#     count_all = len(papers)
#     partial_processing_peersum = functools.partial(processsing_peersum_multi_process, papers=papers, nlp=nlp,
#                                                   lemmatization=lemmatization,
#                                                   removing_stop_words=removing_stop_words, sentence_split=sentence_split)
#     with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
#         papers_processed = list(tqdm(p.imap(partial_processing_peersum, range(count_all), chunksize=64), total=count_all, desc="pre_processing peersum"))
#     return papers_processed
#
# if __name__=="__main__":
#     with open("peersum.json", "r") as f:
#         papers = json.load(f)
#
#     processing_peersum(papers, False, False, True)
#     print("peersum_cleaned.json")
#
#     processing_peersum(papers, True, False, True)
#     print("peersum_lemma.json")
#
#     processing_peersum(papers, True, True, True)
#     print("peersum_lemma_stop.json")


# with open("novel_ngrams_mds.out") as f:
#     lines = f. readlines()
# for line in lines:
#     if "Novel 2-grams" in line:
#         print(line)
#     if "Novel 3-grams" in line:
#         print(line)
#     if "Novel 4-grams" in line:
#         print(line)

# print(multiprocessing.cpu_count())


# rouge = Rouge(metrics=["rouge-1", "rouge-2"])

# source_documents_train, summaries_train, source_documents_val, summaries_val, source_documents_test, summaries_test = loading_mds("../../preparing_data/", "wcep_lemma_stop")
# summaries = summaries_train+summaries_val+summaries_test
# source_documents = source_documents_train+source_documents_val+source_documents_test
# cluster = 0
# for s, c in zip(summaries[4507:], source_documents[4507:]):
#     for d in c:
#         # s = re.sub('\s+', ' ', s)
#         # d = re.sub('\s+', ' ', d)
#         print("summary", s)
#         print("Source document", d)
#         scores = rouge.get_scores(d, s)[0]
#         print("wcep", cluster, scores["rouge-1"]["r"])
#         print("wcep", cluster, scores["rouge-2"]["r"])
#         # print("wcep", cluster, scores["rouge-l"]["f"])
#     cluster += 1
#
#
#
# source_documents_train, summaries_train, source_documents_val, summaries_val, source_documents_test, summaries_test = loading_mds("../../preparing_data/", "multinews_lemma_stop")
# summaries = summaries_train+summaries_val+summaries_test
# source_documents = source_documents_train+source_documents_val+source_documents_test
# cluster = 0
# for s, c in zip(summaries[10102:], source_documents[10102:]):
#     for d in c:
#         s = re.sub('[\s]+', '\s', s)
#         d = re.sub('[\s]+', '\s', d)
#         print("summary", s)
#         print("Source document", d)
#
#         scores = rouge.get_scores(d, s)[0]
#         print("multinews", cluster, scores["rouge-1"]["r"])
#         print("multinews", cluster, scores["rouge-2"]["r"])
#         print("multinews", cluster, scores["rouge-l"]["f"])
#     cluster += 1

# h = ""
# r1 = "I am a kit."
# r2 = "This a nice kit"
# print(h.strip()=="")
# print(rouge.get_scores(h, r1, ignore_empty=True))
# print(rouge.get_scores(h, r1))
# print(rouge.get_scores(h, r2))

# s = [0.0]*10
#
# for i in range(11):
#     index = i%len(s)
#     s[index] = 2


# from preparing_data.data_loader import loading_mds
# from peersum.pre_processing.data_loader import loading_peersum

# pr = [1, 3, 4]
# tr = [5, 6]
# x =4
#
# def change(pr, tr, x):
#     pr[0] = 0
#     tr[1] = 0
#     x+=1
#     return pr, tr, x
#
# print(change(pr+tr, tr, 4))
# print(pr, tr, x)
#
#
# import multiprocessing
# print(multiprocessing.cpu_count())



a = {"a":1}
a.update({"a":2, "b":1})
print(a)
