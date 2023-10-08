from datasets import load_dataset

peersum = load_dataset('json', data_files='../../datasets/peersum.json', split='all')
print("all data", len(peersum))

target_id = "iclr_2022"
count = 0
for item in peersum:
    if item["paper_id"].startswith(target_id):
        count += 1
print(target_id, count)