import jsonlines
import os

n_clusters = 10

dir = "./splitted_data_part2"
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

samples = []
with jsonlines.open("peersum_sampled_part2.json") as reader:
    for sample in reader:
        samples.append(sample)
print("all samples", len(samples))

for x in range(n_clusters):
    count = int(len(samples) / n_clusters)
    samples_cluster = samples[count * x:count * (x + 1)]
    with jsonlines.open(dir + "/peersum_sampled_%d.json" % x, "w") as writer:
        writer.write_all(samples_cluster)
