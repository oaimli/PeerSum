# cleaning texts
import spacy
import re
import functools
import multiprocessing
from tqdm import tqdm


def initializing_spacy():
    nlp = spacy.load("en_core_web_sm")
    return nlp


def cleaning_document(document, nlp=None, lemmatization=False, removing_stop_words=False, sentence_split=False):
    # truncate long document to 100000 characters, which needs large scale memory
    if len(document) > 100000:
        document = document[:100000]
    if document.strip() != "":
        # translate Chinese characters into English characters
        table = {ord(f): ord(t) for f, t in zip(u'，。！？【】（）％＃＠＆１２３４５６７８９０', u',.!?[]()%#@&1234567890')}
        document = document.translate(table).lower()

        # remove special punctuations
        document = re.sub('[^a-z,.!?;():\s]+', '', document)

        # replace continued punctuations
        document = re.sub('[,]+', ',', document)
        document = re.sub('[.]+', '.', document)
        document = re.sub('[!]+', '!', document)
        document = re.sub('[?]+', '?', document)
        document = re.sub('[;]+', ';', document)
        document = re.sub('[:]+', ':', document)
        document = re.sub('[\s]+', ' ', document)
        # document = document.replace("\n", " ")
        # document = " ".join(document.split())

        # tokenization, lemmatization and stop words remove
        if nlp==None:
            nlp = initializing_spacy()


        doc = nlp(document)
        if not sentence_split==True:
            tokens = []
            for token in doc:
                tar = token.lemma_ if lemmatization==True else token.text
                if removing_stop_words==True:
                    if not token.is_stop:
                        tokens.append(tar)
                else:
                    tokens.append(tar)
            document = " ".join(tokens)
        else:
            sents = []
            for sent in doc.sents:
                tokens = []
                for token in sent:
                    tar = token.lemma_ if lemmatization == True else token.text
                    if removing_stop_words == True:
                        if not token.is_stop:
                            tokens.append(tar)
                    else:
                        tokens.append(tar)
                sents.append(" ".join(tokens))
            document = " sentence_split ".join(sents)
    return document


def cleaning_documents_multi_process(i, documents, nlp, lemmatization, removing_stop_words, sentence_split):
    document = documents[i]
    return cleaning_document(document, nlp, lemmatization, removing_stop_words, sentence_split)


def cleaning_documents(documents, multi_process=False, nlp=None, lemmatization=True, removing_stop_words=True, sentence_split=False):
    if nlp==None:
        nlp = initializing_spacy()

    if multi_process==False:
        document_cleaned = []
        for d in documents:
            d = cleaning_document(d, nlp, lemmatization, removing_stop_words, sentence_split)
            if d.strip()!="":# this ensures that each source document is not null
                document_cleaned.append(d)
        return document_cleaned
    else:
        count_all = len(documents)
        partial_cleaning_documents = functools.partial(cleaning_documents_multi_process, documents=documents, nlp=nlp,
                                                      lemmatization=lemmatization,
                                                      removing_stop_words=removing_stop_words, sentence_split=sentence_split)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            document_cleaned = list(
                tqdm(p.imap(partial_cleaning_documents, range(count_all), chunksize=64), total=count_all,
                     desc="cleaning documents"))
        return document_cleaned



if __name__ == "__main__":
    texts = ["I an42 Self is 3 （。 & = Jim looked."]
    a = 3 if 4>5 else 6
    print(a)
    print(cleaning_documents(texts))
    print(len("dd  d"))
    print("dd daaa"[:3])

    table = {ord(f): ord(t) for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０',
        u',.!?[]()%#@&1234567890')}
    t = u'中国，中文，标点符号！你好？１２３４５＠＃【】+=-（）adf dd dddd   d'
    t2 = t.translate(table)
    print(t2, cleaning_document("a, bd, cddw. dww!!!! #\nddddd 3334")+ "hh")
    print(cleaning_document("a, bd, cddw. dww!!!! #\nddddd 3334").split("\n"))

