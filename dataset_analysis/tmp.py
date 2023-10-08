import jsonlines

for dataset in ["multinews", "wcep", "multixscience", "wikisum"]:
    print(dataset)
    papers = []
    with jsonlines.open("../../datasets/%s_lemma_stop.json"%dataset) as reader:
        for paper in reader:
            paper["summary"] = paper["summary"].replace(" sentence_split", "")
            source_documents = paper["source_documents"]
            source_documents_new = []
            for d in source_documents:
                source_documents_new.append(d.replace(" sentence_split", ""))
            paper["source_documents"] = source_documents_new
            papers.append(paper)
    with jsonlines.open("processed/%s_processed.json"%dataset, "w") as writer:
        writer.write_all(papers)