import os

def transforming_snli(snli_dir):
    with open(os.path.join(snli_dir, "snli_1.0_test.txt")) as f:
        lines = f.readlines()
    sentence_pairs = []
    sentence_pairs.append("index" + "\t" + "sentence_1" + "\t" + "sentence_2" + "\t" + "gold_label" + "\t" + "cluster_index" + "\n")

    index = 0
    for line_index, line in enumerate(lines):
        line = line.split("\t")
        if line_index == 0:
            continue
        if line[0] in ["contradiction", "entailment", "neutral"]:
            sentence_pairs.append(str(index) + "\t" + line[5] + "\t" + line[6] + "\t" + line[0] + "\t" + "0" + "\n")
            index += 1
    return sentence_pairs


if __name__ == "__main__":
    sentence_pairs = transforming_snli("glue_data/snli_1.0")
    output_dir = "glue_data/SNLI"
    print("all sentence pairs", len(sentence_pairs))
    with open(os.path.join(output_dir, "test.tsv"), "w") as f:
        f.writelines(sentence_pairs)