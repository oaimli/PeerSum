import requests
import json
import os
import pdfplumber
import pdfminer.pdfparser

def download_pdf(pdf_url, folder, year):
    if pdf_url.endswith(".pdf"):
        file_name = folder + year + "_" + pdf_url.split("/")[-1]
        if not os.path.exists(file_name):
            response = requests.get(url=pdf_url,stream="TRUE")
            with open(file_name,'wb') as file:
                for data in response.iter_content():
                    file.write(data)
            print(pdf_url)
        else:
            print(file_name, "exists")
    else:
        print("NOT PDF", pdf_url)


def get_responses(response_file):
    content = ""
    try:
        with pdfplumber.open(response_file) as pdf:
            for i in range(len(pdf.pages)):
                page = pdf.pages[i]
                page_content = page.extract_text()# delete the page number
                content = content + page_content
    except pdfminer.pdfparser.PDFSyntaxError as e:
        print("PDFSyntaxError")
    return content

if __name__ == "__main__":
    base_url = "https://proceedings.neurips.cc"
    paper_folder = "paper_pdf/"
    response_folder = "response_pdf/"
    years = ["2019", "2020"]


    for year in years:
        with open('data/nips_%s.json'%year, 'r') as f:
            paper_list = json.load(f)

        for paper in paper_list:
            if "pdf" not in paper.keys():
                print(paper["link"])

            if "author_responses" in paper.keys():
                download_pdf(base_url + paper["author_responses"]["pdf"], response_folder, year)
                response_file = response_folder + year + "_" + paper["author_responses"]["pdf"].split("/")[-1]
                if os.path.exists(response_file) and response_file.endswith(".pdf"):
                    if paper["author_responses"].get("responses", default="")=="":
                        paper["author_responses"]["responses"] = get_responses(response_file)
                else:
                    print(response_file, "not exists")


        with open("data/nips_%s.json" % year, "w") as f:
            f.write(json.dumps(paper_list))