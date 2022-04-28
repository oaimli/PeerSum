import json
import random
data = "peersum.json"
with open(data) as f:
    papers = json.load(f)

print("all papers", len(papers))

writers = set([])
for p in papers:
    for r in p["reviews"]:
        if "writer" not in r.keys():
            print(p)
            print(r)
            print(p["paper_id"], r.keys())
        else:
            writers.add(r["writer"])
print(writers)