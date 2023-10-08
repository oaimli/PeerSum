# Summarizing Multiple Documents with Conversational Structure for Meta-review Generation
[![dataset](https://img.shields.io/badge/dataset-%20PeerSum-orange)](https://drive.google.com/drive/folders/1SGYvxY1vOZF2MpDn3B-apdWHCIfpN2uB?usp=sharing) [![arXiv](https://img.shields.io/badge/arxiv-2305.01498-lightgrey)](https://arxiv.org/abs/2305.01498)

### Overview
Text summarization systems need to recognize internal relationships among source texts and effectively aggregate and process information from them to generate high-quality summaries. It is particularly challenging in multi-document summarization (MDS) due to the complexity of the relationships among (semi-)parallel source documents. However, existing MDS datasets do not provide explicit inter-document relationships among the source documents, and this makes it hard to research inter-document relationship comprehension of abstractive summarization models. To address this, we present PeerSum, a novel dataset for generating meta-reviews of scientific papers. The meta-reviews can be interpreted as abstractive summaries of reviews, multi-turn discussions among reviewers and the paper author, and the paper abstract. These source documents have rich inter-document relationships with an explicit hierarchical conversational structure, cross-references and (occasionally) conflicting information. 
![image](https://github.com/oaimli/PeerSum/assets/12547070/93a28a56-7100-43ee-baa8-d84f5fca135d)


To introduce the structural inductive bias into pre-trained language models, we introduce \textbf{\mram} (\underline{R}elationship-\underline{a}ware \underline{M}ulti-task \underline{Me}ta-review Generato\underline{r}), a model that uses sparse attention based on the conversational structure and a multi-task training objective that predicts metadata features (e.g., review ratings).
Our experimental results show that \mram outperforms other strong baseline models in terms of a suite of automatic evaluation metrics. Further analyses, however, reveal that \mram and other models struggle to handle conflicts in source documents of \ps, suggesting meta-review generation is a challenging task and a promising avenue for further research.\footnote{The dataset and code are available at \url{https://github.com/oaimli/PeerSum}}

PeerSum is constructed based on peer reviews in [OpenReview](https://openreview.net/). For more details please refer to our paper. Please feel free to use our dataset with a citation. You can download them from [Google Drive](https://drive.google.com/drive/folders/1SGYvxY1vOZF2MpDn3B-apdWHCIfpN2uB?usp=sharing) or .

## Updates
* The paper is accepted at Findings of EMNLP 2023 (Soundness: 3, 3, 4; Excitement: 3, 4, 4; Confidence:3, 3, 4), October 8, 2023.
* More data added for NeurIPS 2022, April 20, 2023. 
* Crawled more data for ICLR 2022 and NeurIPS 2021, February 20, 2022. 
* Initialized the dataset of PeerSum, November 12, 2021.

## Dataset details
Usually, in a multi-document summarization dataset, there are summaries and source documents. In PeerSum, we have different threads of comments started by official reviewers, public readers, or authors themselves as the source documents and the meta-review (with an acceptance outcome) as the ground truth summary. Each sample of this dataset contains a summary, corresponding source documents and also other completementary informtion (e.g., review scores and confidences) for one academic paper. Up to now, the dataset has 14,993 samples (train/validation/test: 11,995/1,499/1,499).

The dataset is stored in the format of json. There are some other information for scientific peer reviews (e.g., review score for each peer and acceptance for each paper) which could be used not only for summarization but also providing other insights for peer reviewing process. The file named 'peersum_all.json' which contains all data samples in your downloaded folder. For each sample, details are based on following keys with explanation:
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


## How to use the data
To use our raw data, you'd better download 'peersum_all.json' from the shared folder and then load dataset with jsonlines.
```python
import jsonlines
peersum = []
with jsonlines.open("peersum_all.json") as reader:
    for line in reader:
        peersum.append(line)
```

If you are only interested in summarization (generating the meta-review automatically in our paper). You could load it directly with the [Dataset](https://huggingface.co/datasets/oaimli/PeerSum) library by Huggingface as follows (some attributes are removed):
```python
from datasets import load_dataset
peersum_all = load_dataset('oaimli/PeerSum', split='all')
peersum_train = dataset_all.filter(lambda s: s['label'] == 'train')
peersum_val = dataset_all.filter(lambda s: s['label'] == 'val')
peersum_test = dataset_all.filter(lambda s: s['label'] == 'test')
```


## What are in this Repository
The code will be updated soon.
```
/
├── analysis/               --> (Code for dataset analysis and comparison)
├── crawling_data/          --> (Code for crawling data from the websites)
├── dataset/                --> (A foler containing the dataset)
├── other/                  --> (Other information like the data example)
├── plot/                   --> (Plot the results)
├── preparing_data/         --> (Code for data pre-processing)   
└── README.md               --> (The readme file)
```

If you are going to use our dataset in your work, please cite our paper:

[Li et al. 2023] Miao Li, Eduard Hovy, and Jey Han Lau. "Towards Summarizing Multiple Documents with Hierarchical Relationships". arXiv, 2023.
```
@inproceedings{peersum_2023,
  title={Towards Summarizing Multiple Documents with Hierarchical Relationships},
  author={Miao Li, Eduard Hovy, and Jey Han Lau},
  booktitle={Arxiv},
  year={2023}
}
```



