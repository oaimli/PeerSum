import jsonlines

part1 = []
with jsonlines.open("peersum_sampled_part1.json") as reader:
    for line in reader:
        part1.append(line)

part2 = []
with jsonlines.open("peersum_sampled_part2.json") as reader:
    for line in reader:
        part2.append(line)

all_samples_source = set([])
all_samples_summary = set([])
for sample in part1 + part2:
    all_samples_source.add(sample["source_documents"][0])
    all_samples_summary.add(sample["summary"])
print(len(all_samples_source))
print(len(all_samples_summary))