import jsonlines
import numpy as np

data = []
with jsonlines.open('../../../peersum/crawling_data/data/peersum_all.json') as reader:
    for line in reader:
        data.append(line)
print("all data", len(data))

heights = []
widths = []
papers_errors = set([])
for sample in data:
    paper_id = sample["paper_id"]
    reviews = sample["reviews"]

    review_ids = []
    review_ids.append(sample["paper_id"])
    for review in reviews:
        review_ids.append(review["review_id"])

    reviews_tree = []
    for review in reviews:
        if review["reply_to"] in review_ids:
            reviews_tree.append(review)
        else:
            print(paper_id)
            papers_errors.add(paper_id)
            print(review)

    if len(reviews)-len(reviews_tree)>0:
        print(len(reviews), len(reviews_tree))
    if len(set(review_ids))-1!=len(reviews):
        print("repeated review")

    # height
    current_nodes = []
    for review in reviews_tree:
        if review["reply_to"] == sample["paper_id"]:
            current_nodes.append(review["review_id"])

    height_max = 0
    tmp = current_nodes
    while len(tmp)>0:
        height_max += 1
        tmp = []
        for review in reviews_tree:
            if review["reply_to"] in current_nodes:
                tmp.append(review["review_id"])
        current_nodes = tmp
    # print(sample["paper_id"], height_max)
    heights.append(height_max)

    # width
    leaves = 0
    for review_i in reviews_tree:
        review_i_id = review_i["review_id"]
        is_leaf = True
        for review_j in reviews_tree:
            if review_j["reply_to"] == review_i_id:
                is_leaf = False
                break
        if is_leaf:
            leaves += 1
    widths.append(leaves)

    if sample["paper_id"].startswith("nips"):
        for review in reviews_tree:
            if review["writer"] == "public":
                print(review)


print("error papers", len(papers_errors))
for id in sorted(papers_errors):
    print(id)
print("average max height", np.mean(heights), np.std(heights))
print("average max width (leaves)", np.mean(widths), np.std(widths))