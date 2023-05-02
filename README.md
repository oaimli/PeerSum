# Towards Summarizing Multiple Documents with Hierarchical Relationships
[![dataset](https://img.shields.io/badge/dataset-%20PeerSum-orange)](https://drive.google.com/drive/folders/1SGYvxY1vOZF2MpDn3B-apdWHCIfpN2uB?usp=sharing)

### Overview
To enhance the capabilities of MDS systems we present PeerSum, a novel dataset for generating meta-reviews of scientific papers, where the meta-reviews are highly abstractive and genuine summaries of reviews and corresponding discussions. These source documents have rich inter-document relationships of an explicit hierarchical structure with cross-references and often feature conflicts. As there is a scarcity of research that incorporates hierarchical relationships into MDS systems through attention manipulation on pre-trained language models, we additionally present Rammer (Relationship-aware Multi-task Meta-review Generator), a meta-review generation model that uses sparse attention based on the hierarchical relationships and a multi-task objective that predicts several metadata features in addition to the standard text generation objective. Our experimental results show that PeerSum is a challenging dataset, and Rammer outperforms other strong baseline MDS models under various evaluation metrics. 

PeerSum is constructed based on peer reviews in [the Openreview system](https://openreview.net/). More details please refer to our paper. Please feel free to use our dataset with a citation. You can download them from [Google Drive]([https://drive.google.com/drive/folders/1M1QhIwjuZOG3QdxNFqY7J5Ik5UsDA0Sk?usp=sharing](https://drive.google.com/drive/folders/1SGYvxY1vOZF2MpDn3B-apdWHCIfpN2uB?usp=sharing)).

## Updates
* crawled more data for ICLR 2022 and NeuIPS 2021, February 20, 2022. 
* uploaded the first version of PeerSum, November 12, 2021.

## Dataset details
Usually, in a multi-document summarization dataset, there are summaries and source documents. In PeerSum, we have different threads of comments started by official reviewers, public readers, or authors themselves as the source documents and the meta-review (with an acceptance outcome) as the ground truth summary. Each sample of this dataset contains a summary, corresponding source documents and also other completementary informtion (e.g., review scores) for one paper. Up to now, the dataset has 15,879 samples.

The dataset is stored in the format of json. There are some other information for scientific peer reviews (e.g., review score for each peer and acceptance for each paper) which could be used not only for summarization but also providing other insights for peer reviewing process. For each sample, details are based on following keys with explanation:
```
*paper_id,
*paper_title,
*paper_abstract,
*paper_score,
*paper_acceptance,
*meta_review,
*official_threads, [[{document_id, writer, comment, replyto}]]
*public_threads, [[{document_id, writer, comment, replyto}]]
*author_threads, [[{document_id, writer, comment, replyto}]]
```
If you are interested in summarization (generating the meta-review automatically). You could load it directly with the Dataset library by Huggingface as follows:
```python
from datasets import load_dataset
peersum_train = load_dataset("oaimli/PeerSum", split="train")
peersum_validation = load_dataset("oaimli/PeerSum", split="validation")
peersum_test = load_dataset("oaimli/PeerSum", split="test")
```


## What are in this Repository
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
