import jsonlines

samples = []
with jsonlines.open("../../crawling_data/data/peersum_all.json") as reader:
    for line in reader:
        samples.append(line)
print("all samples", len(samples))

samples_disagreement = 0
samples_valid = 0
for sample in samples:
    samples_valid += 1
    reviews = sample["reviews"]
    label = sample["label"]
    with_disagreement = False
    for i, review_i in enumerate(reviews):
        for j, review_j in enumerate(reviews):
            if j > i and review_i["rating"]!=-1 and review_j["rating"]!=-1:
                dis = review_i["rating"] - review_j["rating"]
                if abs(dis) >= 4:
                    with_disagreement = True
    if with_disagreement:
        samples_disagreement += 1
        # print(sample)
print(samples_disagreement)
print("disgreement ratio", samples_disagreement/samples_valid)
