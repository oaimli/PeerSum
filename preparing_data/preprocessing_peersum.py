# cleaning summaries and source documents, cleaning, lemmatization, stop words.
import sys
import json
import multiprocessing
import functools
from tqdm import tqdm
import spacy

sys.path.append("../../")
from utils.cleaning import cleaning_document, initializing_spacy


def processsing_peersum_multi_process(i, papers, nlp, lemmatization, removing_stop_words, sentence_split):
    paper = papers[i]

    paper["meta_review"] = cleaning_document(paper["meta_review"], nlp, lemmatization, removing_stop_words, sentence_split)

    for review in paper["reviews"]:
        review_text = review["content"]["comment"]
        review["content"]["comment"] = cleaning_document(review_text, nlp, lemmatization, removing_stop_words, sentence_split)
    return paper


def processing_peersum(papers, lemmatization=True, removing_stop_words=True, sentence_split=True):
    nlp = initializing_spacy()
    count_all = len(papers)
    partial_processing_peersum = functools.partial(processsing_peersum_multi_process, papers=papers, nlp=nlp,
                                                  lemmatization=lemmatization,
                                                  removing_stop_words=removing_stop_words, sentence_split=sentence_split)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        papers_processed = list(tqdm(p.imap(partial_processing_peersum, range(count_all), chunksize=64), total=count_all, desc="pre_processing peersum"))
    return papers_processed

if __name__=="__main__":
    with open("../dataset/peersum_new.json", "r") as f:
        papers = json.load(f)

    papers_new = processing_peersum(papers, False, False, True)
    with open("../dataset/peersum_new_cleaned.json", "w") as f:
        f.write(json.dumps(papers_new))

    papers_new = processing_peersum(papers, True, False, True)
    with open("../dataset/peersum_new_lemma.json", "w") as f:
        f.write(json.dumps(papers_new))

    papers_new = processing_peersum(papers, True, True, True)
    with open("../dataset/peersum_new_lemma_stop.json", "w") as f:
        f.write(json.dumps(papers_new))