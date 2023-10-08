"""
Evaluating generated summaries in terms of different metrics

"""
import json
import os
from metrics import evaluating_summaries_multi_sources

if __name__ == "__main__":
    dataset_name = "peersum"
    generated_summaries_folder = ""

    files = os.listdir(generated_summaries_folder)
    print(dataset_name, "all samples", len(files))

    references = []
    predictions = []
    document_clusters = []
    for file in files:
        if not os.path.isdir(file):
            with open(os.path.join(generated_summaries_folder, file)) as f:
                result = json.load(f)
                # print(result)
                if "source_documents" in result.keys():
                    source_documents = result["source_documents"]
                    document_clusters.append(source_documents)
                predictions.append(result["prediction"])
                references.append(result["reference"])

    print(evaluating_summaries_multi_sources(gold_summaries=references, generated_summaries=predictions, source_document_clusters=document_clusters))
