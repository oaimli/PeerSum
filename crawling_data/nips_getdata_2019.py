from bs4 import BeautifulSoup
import urllib.request
import json
import os
from urllib.error import HTTPError

base_url = "https://proceedings.neurips.cc"
year_url= base_url + "/paper/2019"
output_file_name = "../data/nips_2019.json"

req=urllib.request.Request(year_url)
resp=urllib.request.urlopen(req)
data=resp.read().decode('utf-8')

papers = []
soup = BeautifulSoup(data, 'html.parser')
paper_list = soup.find("div", attrs={'class':'col'}).ul.find_all('li')
print(len(paper_list))

for item in paper_list:
    paper = {}
    paper["link"] = item.a["href"]
    paper["title"] = item.a.get_text()
    paper["authors"] = item.i.get_text()

    req = urllib.request.Request(base_url + paper["link"])
    try:
        resp = urllib.request.urlopen(req)
        data = resp.read().decode('utf-8')
        soup = BeautifulSoup(data, 'html.parser')
        content = soup.find("div", attrs={'class': 'col'})

        hrefs = content.div.find_all("a")
        for h in hrefs:
            hcontent = h.get_text()
            # meta-review
            if hcontent=="MetaReview »":
                req = urllib.request.Request(base_url + h["href"])
                # print(base_url + content.div.find_all("a")[3].get_text())
                resp = urllib.request.urlopen(req)
                data = resp.read()
                soup = BeautifulSoup(data, 'html.parser')
                paper["meta_review"] = soup.find_all("div")[-1].get_text()

            # reviews
            if hcontent == "Reviews »":
                req = urllib.request.Request(base_url + h["href"])
                resp = urllib.request.urlopen(req)
                data = resp.read()
                soup = BeautifulSoup(data, 'html.parser')
                paper["reviews"] = []
                for review in soup.find_all("pre", attrs={"class":"review"}):
                    paper["reviews"].append(review.find_next().get_text())

            if hcontent == "Paper":
                paper["pdf"] = h["href"]


        # abstract
        paper["abstract"] = content.find_all('p')[-2].get_text()

        paper["comment"] = "accept"
    except HTTPError as e:
        print("URL error")

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