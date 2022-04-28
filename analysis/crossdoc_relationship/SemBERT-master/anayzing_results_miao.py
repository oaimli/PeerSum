import os
import random


data_name = "WIKISUM"
data_dir = "glue_data/%s/"%data_name
with open("snli_model_dir/%s_pred_results.tsv"%data_name) as f:
    lines = f.readlines()

predictions = []
for line in lines:
    predictions.append(line.strip().split("\t")[-1])

with open(os.path.join(data_dir, "test.tsv")) as f:
    lines = f.readlines()
eval_examples = []
for line_index, line in enumerate(lines):
    if line_index>0:
        line_split = line.split("\t")
        if line_split[3]=="contradiction":
            eval_examples.append([line_split[1], line_split[2]])


count_all = len(predictions)
indexes = range(count_all)
target_indexes = random.sample(indexes, 20)
for target in target_indexes:
    exam = eval_examples[target]
    print(exam[0], exam[1], "contradiction")