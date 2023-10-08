# Summarizing Multiple Documents with Conversational Structure for Meta-review Generation
[![dataset](https://img.shields.io/badge/dataset-%20PeerSum-orange)](https://huggingface.co/datasets/oaimli/PeerSum) [![arXiv](https://img.shields.io/badge/arxiv-2305.01498-lightgrey)](https://arxiv.org/abs/2305.01498)

### Overview
Text summarization systems need to recognize internal relationships among source texts and effectively aggregate and process information from them to generate high-quality summaries. It is particularly challenging in multi-document summarization (MDS) due to the complexity of the relationships among (semi-)parallel source documents. However, existing MDS datasets do not provide explicit inter-document relationships among the source documents, and this makes it hard to research inter-document relationship comprehension of abstractive summarization models. To address this, we present PeerSum, a novel dataset for generating meta-reviews of scientific papers. The meta-reviews can be interpreted as abstractive summaries of reviews, multi-turn discussions among reviewers and the paper author, and the paper abstract. These source documents have rich inter-document relationships with an explicit hierarchical conversational structure, cross-references and (occasionally) conflicting information, as shown in Fig. 1. 
![image](https://github.com/oaimli/PeerSum/assets/12547070/aa23aa8a-5fed-4cd7-a025-852dbeb0bfdc)

PeerSum features a hierarchical conversational structure among the source documents (which includes the reviews, responses and the paper abstract in different threads). PeerSum has several distinct advantages over existing MDS datasets: 
- We show that the meta-reviews are largely faithful to the corresponding source documents despite being highly abstractive;
- The source documents have rich inter-document relationships with an explicit conversational structure;
- The source documents occasionally feature conflicts which the meta-review needs to handle as reviewers may have a disagreement on reviewing a scientific paper, and we explicitly provide indicators of conflict relationships along with the dataset;
- It has a rich set of metadata, such as review rating/confidence and paper acceptance outcome, the latter of which can be used for assessing the quality of automatically generated meta-reviews.

To introduce the structural inductive bias into pre-trained language models, we introduce Rammer (Relationship-aware Multi-task Meta-review Generator), a model that uses sparse attention based on the conversational structure and a multi-task training objective that predicts metadata features (e.g., review ratings). Our experimental results show that our model outperforms other strong baseline models in terms of a suite of automatic evaluation metrics. Further analyses, however, reveal that our model and other models struggle to handle conflicts in source documents of PeerSum, suggesting meta-review generation is a challenging task and a promising avenue for further research. For more details about the model and experiments, please refer to our paper.

## Updates
* The paper is accepted at Findings of EMNLP 2023 (Soundness: 3, 3, 4; Excitement: 3, 4, 4. October 8, 2023.
* More data added for NeurIPS 2022. April 20, 2023. 
* Crawled more data for ICLR 2022 and NeurIPS 2021. February 20, 2022. 
* Initialized the dataset of PeerSum. November 12, 2021.

## Dataset details
PeerSum is constructed based on peer reviews crawled from [OpenReview](https://openreview.net/). Usually, in an MDS dataset, there are summaries and source documents. In PeerSum, we have different comments started by official reviewers, public readers, or authors themselves as the source documents and the meta-review (with an acceptance outcome) as the ground truth summary. Each sample of this dataset contains a summary, corresponding source documents and also other completementary informtion (e.g., review ratings and confidences) for one academic paper. For more details please refer to our [paper](https://arxiv.org/abs/2305.01498). Please feel free to use our dataset with a citation. You can download the raw data from [Google Drive](https://drive.google.com/drive/folders/1SGYvxY1vOZF2MpDn3B-apdWHCIfpN2uB?usp=sharing) and cleaned summarization data from [Huggingface](https://huggingface.co/datasets/oaimli/PeerSum).

The raw dataset is stored in the format of JSON. There is some other information for scientific peer reviews (e.g., review rating for each peer review and acceptance for each paper) which could be used not only for summarization but also providing other insights for the peer reviewing process. The file, named 'peersum_all.json', contains all data samples in your downloaded folder. For each sample, details are based on the following keys with explanations:
```
* paper_id: str
* paper_title: str
* paper_abstract, str
* paper_acceptance, str
* meta_review, str
* reviews, [{review_id, writer, comment, rating, confidence, reply_to}] (all reviews and discussions)
* label, str, (train, val, test)

* Please note:
* confidence: 1-5, int
* rating: 1-10, int
* writer: str, (author, official_reviewer, public)
```

The Huggingface dataset is mainly for multi-document summarization. It contains the same number of samples and each sample comprises information with the following keys with explanations:
```
* paper_id: str (a link to the raw data)
* paper_title: str
* paper_abstract, str
* paper_acceptance, str
* meta_review, str
* review_ids, list(str)
* review_writers, list(str)
* review_contents, list(str)
* review_ratings, list(int)
* review_confidences, list(int)
* review_reply_tos, list(str)
* label, str, (train, val, test)
```

You can access the original webpage of the paper via the link https://openreview.net/forum?id= with paper_id. For example, the paper is iclr_2018_Hkbd5xZRb, then the link is [https://openreview.net/forum?id=Hkbd5xZRb](https://openreview.net/forum?id=Hkbd5xZRb).

Up to now, the dataset has in total of 14,993 samples (train/validation/test: 11,995/1,499/1,499) from the following conferences. (You can use paper_id to filter samples in different years and venues. We will include more data from coming conferences in future.)
```
* NeurIPS: 2021, 2022
* ICLR: 2018, 2019, 2020, 2021, 2022
```


## How to use the data
To use our raw data, you'd better download '[peersum_all.json](https://drive.google.com/file/d/1XCF4omItvv-cyUkLhzt-DLDkg2AKga2O/view?usp=drive_link)' from the shared folder and then load the dataset with jsonlines.
```python
import jsonlines
peersum = []
with jsonlines.open("peersum_all.json") as reader:
    for line in reader:
        peersum.append(line)
```

If you are only interested in summarization (generating the meta-review automatically in our paper). You could load [PeerSum](https://huggingface.co/datasets/oaimli/PeerSum) directly with the datasets library by Huggingface as follows:
```python
from datasets import load_dataset
peersum_all = load_dataset('oaimli/PeerSum', split='all')
peersum_train = peersum_all.filter(lambda s: s['label'] == 'train')
peersum_val = peersum_all.filter(lambda s: s['label'] == 'val')
peersum_test = peersum_all.filter(lambda s: s['label'] == 'test')
```


## What are in this Repository
```
/
├── acceptance_prediction/  --> (Code for predicting paper acceptance with generated summaries)
├── crawling_data/          --> (Code for crawling data from the websites and preparing training datasets, Section 3 in the paper)
├── dataset/                --> (A folder containing the dataset)
├── dataset_analysis/       --> (Code for data quality analysis, Section 3 in the paper)
├── human_annotation/       --> (Annotation results for the dataset quality in Section 3.3 and evaluation of generated results in Section 5.3)
├── loading_data/           --> (Code for loading dataset into the models)
├── mtsum_meta/             --> (The model for the Relationship-aware Multi-task Meta-review Generator)   
├── other/                  --> (Other information like the data example)
├── plot/                   --> (Plot the results)
├── utils/                  --> (Code for automatic evaluation and data pre-processing)   
└── README.md               --> (This readme file)
```

If you are going to use our dataset in your work, please cite our paper:

[Li et al. 2023] Miao Li, Eduard Hovy, and Jey Han Lau. "Summarizing Multiple Documents with Conversational Structure for Meta-review Generation". arXiv, 2023.
```
@inproceedings{peersum_2023,
  title={Summarizing Multiple Documents with Conversational Structure for Meta-review Generation},
  author={Miao Li, Eduard Hovy, and Jey Han Lau},
  booktitle={Findings of EMNLP 2023},
  year={2023}
}
```



