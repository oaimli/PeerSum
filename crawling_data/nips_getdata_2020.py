from bs4 import BeautifulSoup
import urllib.request
import json
import os

base_url = "https://proceedings.neurips.cc"
year_url= base_url + "/paper/2020"
output_file_name = "../data/nips_2020.json"

req=urllib.request.Request(year_url)
resp=urllib.request.urlopen(req)
data=resp.read().decode('utf-8')

papers = []
soup = BeautifulSoup(data, 'html.parser')
paper_list = soup.find("div", attrs={'class':'col'}).ul.find_all('li')
print(len(paper_list))

exist_papers = []
if os.path.isfile(output_file_name):
    with open(output_file_name, 'r') as f:
        exist_papers = json.load(f)
print("exist papers", len(exist_papers))

for item in paper_list[len(exist_papers):]:
    paper = {}
    paper["link"] = item.a["href"]
    paper["title"] = item.a.get_text()
    paper["authors"] = item.i.get_text()

    req = urllib.request.Request(base_url + paper["link"])
    resp = urllib.request.urlopen(req)
    data = resp.read().decode('utf-8')
    soup = BeautifulSoup(data, 'html.parser')
    content = soup.find("div", attrs={'class': 'col'})

    hrefs = content.div.find_all("a")
    for h in hrefs:
        hcontent = h.get_text()

        # meta-review
        if hcontent == "MetaReview »":
            req = urllib.request.Request(base_url + h["href"])
            # print(base_url + content.div.find_all("a")[3].get_text())
            resp = urllib.request.urlopen(req)
            data = resp.read()
            soup = BeautifulSoup(data, 'html.parser')
            paper["meta_review"] = soup.p.get_text()

        # reviews
        if hcontent == "Review »":
            req = urllib.request.Request(base_url + h["href"])
            resp = urllib.request.urlopen(req)
            data = resp.read()
            soup = BeautifulSoup(data, 'html.parser')
            paper["reviews"] = []
            for tmp in soup.find_all("h3")[1:]:
                review = {}
                sc = tmp.find_next("p")
                review["summary_and_contributions"] = sc.get_text()
                st = sc.find_next("p")
                review["strengths"] = st.get_text()
                we = st.find_next("p")
                review["weaknesses"] = we.get_text()
                co = we.find_next("p")
                review["correctness"] = co.get_text()
                cl = co.find_next("p")
                review["clarify"] = cl.get_text()
                rt = cl.find_next("p")
                review["relation_to_prior_work"] = rt.get_text()
                re = rt.find_next("p")
                review["reproducibility"] = re.get_text()
                af = re.find_next("p")
                review["additional_feedback"] = af.get_text()
                paper["reviews"].append(review)

        if hcontent == "Paper »":
            paper["pdf"] = h["href"]

        # author response
        if hcontent == "AuthorFeedback »":
            author_responses = {}
            author_responses["pdf"] = h["href"]
            paper["author_responses"] = author_responses

    # abstract
    paper["abstract"] = content.find_all('p')[-2].get_text()
    paper["comment"] = "accept"
    papers.append(paper)

    if len(papers)%50==0:
        exist_papers = []
        if os.path.isfile(output_file_name):
            with open(output_file_name, 'r') as f:
                exist_papers = json.load(f)

        exist_papers.extend(papers)
        f = open(output_file_name, 'w')
        f.write(json.dumps(exist_papers))
        f.close()
        print("paper count", len(exist_papers))

        papers = []

with open(output_file_name, 'r') as f:
    exist_papers = json.load(f)

exist_papers.extend(papers)
f = open(output_file_name, 'w')
f.write(json.dumps(exist_papers))
f.close()
print("paper count", len(exist_papers))

print("Done")