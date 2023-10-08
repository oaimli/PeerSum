# loading mds data after downloading datasets stored in a same format
import os
import jsonlines


def loading_mds(folder, data_name):
    print("loading mds", data_name)

    samples = []
    file_name = folder + "/%s.json"%data_name
    if os.path.isfile(file_name):
        with jsonlines.open(file_name) as reader:
            for line in reader:
                samples.append(line)
    else:
        print("File does not exist.")

    return samples


if __name__=="__main__":
    import random
    samples = loading_mds(folder="../../datasets", data_name="wikisum")

    indexes = range(len(samples))
    target_indexes = random.sample(indexes, 5)
    for i in target_indexes:
        sample = samples[i]
        print("*********************")
        for item in sample["source_documents"]:
            print(item)
            print("###########")
        print(sample["summary"])
