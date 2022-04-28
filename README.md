# PeerSum
PeerSum is a multi-document summarization dataset, which is constructed based on peer reviews in [the Openreview system](https://openreview.net/). For further details, please refer to our paper [PeerSum: A Peer Review Dataset for Abstractive Multi-document Summarization](https://arxiv.org/abs/2203.01769). This dataset differs from other MDS datasets (e.g., Multi-News, WCEP, WikiSum, and Multi-XScience) in that our summaries (i.e., the metareviews) are highly abstractive and they are real summaries of the source documents (i.e., the reviews) and it also features disagreements among source documents. Please feel free to use our dataset with a citation. You can download them from [Google Drive](https://drive.google.com/drive/folders/1M1QhIwjuZOG3QdxNFqY7J5Ik5UsDA0Sk?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1fleJ4MXcTQ2PYmlbJ8tDBA?pwd=s3wi).

## Updates
* crawled more data for ICLR 2022 and NeuIPS 2021, and uploaded PeerSum_v2, February 20, 2022. 
* uploaded the first version of PeerSum, November 12, 2021.

## Dataset details
Usually, in a multi-document summarization dataset, there are summaries and source documents. In PeerSum, we have reviews (with scores), comments and responses as the source documents and the meta-review (with an acceptance outcome) as the ground truth summary. Each sample of this dataset contains a summary, corresponding source documents and also other completementary informtion (e.g., review scores) for one paper. Up to now, the second version of PeerSum (peersum_v2) has 16,308 samples, while there are 10,862 samples in the first version.

The dataset is stored in the format of json. For each sample, details are based on following keys with expanation:
```
* paper_id: unique id for each sample
* title: the title of the corresponding paper
* abstract: paper abstract
* score: final score of this paper (if there is not a final, it will be an average of review scores)
* acceptance: acceptance of the paper (e.g., accept, reject or spotlight)
* meta_review: meta-review of the paper and this is treated as the summary
* reviews: [review_id, writer, content (rating, confidence, comment), replyto]   review_id and replyto are for the conversation structure
* label: train, val, test (8/1/1)

For each review (i.e., official review, public comment, or author/reviewer response):
* review_id: unique id of each review
* writer: official_reviewer, public, author
* content: (rating, confidence, comment)
* replyto: connect to a review (review_id and replyto are for the conversation structure)
```

# What are in this Repository


# Citation
If you use this dataset in your projects, please cite our paper:

[Li et al. 2022] Miao Li, Jianzhong Qi, and Jey Han Lau. "PeerSum: A Peer Review Dataset for Abstractive Multi-document Summarization". Arxiv, 2022.

```
@inproceedings{peersum_2022,
  title={PeerSum: A Peer Review Dataset for Abstractive Multi-document Summarization},
  author={Miao Li, Jianzhong Qi, and Jey Han Lau},
  booktitle={Arxiv},
  year={2022}
}
```
