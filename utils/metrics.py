import numpy as np
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

import sys
sys.path.append('../')


def bert_score(candidates, references, scorer=None):
    if scorer == None:
        from bert_score import BERTScorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=True, model_type="microsoft/deberta-xlarge-mnli")
    P, R, F = scorer.score(candidates, references)
    results = {}
    results["p"] = P.mean().item()
    results["r"] = R.mean().item()
    results["f"] = F.mean().item()
    return results


def rouge(reference, candidate, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
          split_summaries=True):
    """
    This is based on rouge-score 0.0.4
    If using rougeLsum, it is necessary to split sentences with '\n' in summaries in advance
    """
    if 'rougeLsum' in types and split_summaries:
        reference = '\n'.join(sent_tokenize(reference))
        candidate = '\n'.join(sent_tokenize(candidate))

    results = {}
    for t in types:
        if t not in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
            print("The type must be selected in rouge1, rouge2, rougeL, and rougeLsum.")
            return results
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=use_stemmer)
    scores = scorer.score(reference, candidate)
    for t in types:
        r = {}
        r["precision"] = scores[t].precision
        r["recall"] = scores[t].recall
        r["fmeasure"] = scores[t].fmeasure
        results[t] = r
    return results


def rouge_corpus(references, candidates, types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True,
                 split_summaries=True):
    if len(references) != len(candidates):
        print("len must be equal")
        return None
    results = {}
    for t in types:
        s = {}
        s['recall'] = []
        s['precision'] = []
        s['fmeasure'] = []
        results[t] = s
    for ref, can in zip(references, candidates):
        s = rouge(ref, can, types=types, use_stemmer=use_stemmer, split_summaries=split_summaries)
        for t in types:
            results[t]['recall'].append(s[t]['recall'])
            results[t]['precision'].append(s[t]['precision'])
            results[t]['fmeasure'].append(s[t]['fmeasure'])

    final_results = {}
    for t in types:
        s = results[t]
        tmp = {}
        tmp['precision'] = np.mean(s['precision'])
        tmp['recall'] = np.mean(s['recall'])
        tmp['fmeasure'] = np.mean(s['fmeasure'])
        final_results[t] = tmp
    return final_results


def bart_score(candidates, references, bart_scorer=None):
    """
    This bart_score cannot be run on A100 GPUs
    To use BARTScore, please download it by running the following command first in this folder
    git clone git@github.com:neulab/BARTScore.git
    """
    if bart_scorer == None:
        from utils.bart_score.bart_score import BARTScorer
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    # generation scores from the first list of texts to the second list of texts.
    return np.mean(bart_scorer.score(candidates, references, batch_size=4))


def unieval(source_documents, references, predictions):
    from utils.unieval.evaluator import get_evaluator
    json_data = []
    for i in range(len(predictions)):
        cur = {}
        cur['system_output'] = predictions[i]
        if source_documents is not None:
            cur['source'] = source_documents[i]
        if references is not None:
            cur['reference'] = references[i]
        json_data.append(cur)
    # Initialize evaluator for a specific task
    evaluator = get_evaluator("summarization")
    # Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(json_data, print_result=False)
    coherences = []
    consistencies = []
    fluencies = []
    relevances = []
    overalls = []
    for eval_score in eval_scores:
        coherences.append(eval_score["coherence"])
        consistencies.append(eval_score["consistency"])
        fluencies.append(eval_score["fluency"])
        relevances.append(eval_score["relevance"])
        overalls.append(eval_score["overall"])
    results = {}
    results["coherence"] = np.mean(coherences)
    results["consistency"] = np.mean(consistencies)
    results["fluency"] = np.mean(fluencies)
    results["relevance"] = np.mean(relevances)
    results["overall"] = np.mean(overalls)
    return results


def evaluating_summaries_single_source(gold_summaries, generated_summaries, source_documents=None):
    score_bert = bert_score(generated_summaries, gold_summaries)
    rouge_results = rouge_corpus(gold_summaries, generated_summaries)
    score_bart = bart_score(generated_summaries, gold_summaries)
    score_unieval = unieval(source_documents=source_documents, references=gold_summaries,
                            predictions=generated_summaries)
    return {"bert_recall": score_bert["r"],
            "bert_precision": score_bert["p"],
            "bert_fmeasure": score_bert["f"],
            "rouge1_fmeasure": rouge_results["rouge1"]["fmeasure"],
            "rouge2_fmeasure": rouge_results["rouge2"]["fmeasure"],
            "rougeLsum_fmeasure": rouge_results["rougeLsum"]["fmeasure"],
            "score_bart": score_bart,
            "unieval_coherence": score_unieval["coherence"],
            "unieval_consistency": score_unieval["consistency"],
            "unieval_fluency": score_unieval["fluency"],
            "unieval_relevance": score_unieval["relevance"],
            "unieval_overall": score_unieval["overall"]
            }


def evaluating_summaries_multi_sources(gold_summaries, generated_summaries, source_document_clusters=None):
    score_bert = bert_score(generated_summaries, gold_summaries)
    rouge_results = rouge_corpus(gold_summaries, generated_summaries)
    score_bart = bart_score(generated_summaries, gold_summaries)
    source_texts = []

    for source_document_cluster in source_document_clusters:
        input_text = []
        max_length_doc = 1024 // len(source_document_cluster)
        for source_document in source_document_cluster:
            # preprocessing
            source_document = source_document.replace("\n", " ")
            source_document = " ".join(source_document.split())

            length = 0
            all_sents = sent_tokenize(source_document)
            for s in all_sents:
                input_text.append(s)
                length += len(s.split())
                if length >= max_length_doc:
                    break
        source_texts.append(" ".join(input_text))
    score_unieval = unieval(source_documents=source_texts, references=gold_summaries,
                            predictions=generated_summaries)
    return {"bert_recall": score_bert["r"],
            "bert_precision": score_bert["p"],
            "bert_fmeasure": score_bert["f"],
            "rouge1_fmeasure": rouge_results["rouge1"]["fmeasure"],
            "rouge2_fmeasure": rouge_results["rouge2"]["fmeasure"],
            "rougeLsum_fmeasure": rouge_results["rougeLsum"]["fmeasure"],
            "score_bart": score_bart,
            "unieval_coherence": score_unieval["coherence"],
            "unieval_consistency": score_unieval["consistency"],
            "unieval_fluency": score_unieval["fluency"],
            "unieval_relevance": score_unieval["relevance"],
            "unieval_overall": score_unieval["overall"]
            }


if __name__ == "__main__":
    print(rouge('Thee quick brownd foxe a. jumps over thee lazy dogr',
                'a Thee quick. brownd foxe dogd jumps over thee log. Thee quick. brown doge jumps on the log.'))
    print(rouge_corpus(['Thee quick brownd foxe a.\n jumps over thee lazy dogr'],
                       [
                           'a Thee quick.\n brownd foxe dogd jumps over thee log. Thee quick. brown doge jumps on the log.']))

    r1 = "police killed the gunman."
    r2 = "the gunman was shot down by police."
    c1 = "police ended the gunman."
    c2 = "the gunman murdered police."

    print(rouge(r1, c1)["rouge1"]["fmeasure"] + rouge(r2, c1)["rouge1"]["fmeasure"])
    print(rouge(r1, c2)["rouge1"]["fmeasure"] + rouge(r2, c2)["rouge1"]["fmeasure"])

    # using bert_score from datasets
    from datasets import load_metric

    scorer = load_metric("rouge", experiment_id="dataset_name")
    s = scorer.compute(
        predictions=[c1],
        references=[r1],
        use_aggregator=False,
        use_stemmer=True,
    )
    print("ROUGE", s)
    print("BERTScore", bert_score([c1, c2], [r1, r2]))
    # using bart_score
    print("BARTScore", bart_score([c1, c2], [r1, r2]))
    # using unieval
    # a list of source documents
    src_list = ['Peter and Elizabeth took a taxi to attend the night party in the city. \
                 While in the party, Elizabeth collapsed and was rushed to the hospital.']
    # a list of human-annotated reference summaries
    ref_list = ['Elizabeth was hospitalized after attending a party with Peter.']
    # a list of model outputs to be evaluataed
    output_list = ['Peter and Elizabeth attend party city. Elizabeth rushed hospital.']
    print("UniEval", unieval(source_documents=src_list, references=ref_list, predictions=output_list))
