import jsonlines

papers = []
papers_with_external_knowledge = []
with jsonlines.open("../../datasets/peersum.json", "r") as reader:
    for paper in reader:
        papers.append(paper)
        summary = paper["summary"]
        if "in my opinion" in summary or "In my opinion" in summary or "read the paper myself" in summary:
            papers_with_external_knowledge.append(paper)

print("Proportion of papers with external knowledge", len(papers_with_external_knowledge)/len(papers))
