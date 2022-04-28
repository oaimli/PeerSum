# # This can just obtained accepted papers.
# from bs4 import BeautifulSoup
# import urllib.request
# import json
# import os
# import openreview
# import time
#
# nips_url = "https://proceedings.neurips.cc"
# year_url= nips_url + "/paper/2021"
# data_folder = "../data/"
# output_file_name = data_folder + "nips_2021.json"
# if not os.path.exists(data_folder):
#     os.mkdir(data_folder)
#     print("data folder created")
#
# req=urllib.request.Request(year_url)
# resp=urllib.request.urlopen(req)
# data=resp.read().decode('utf-8')
#
# papers = []
# soup = BeautifulSoup(data, 'html.parser')
# paper_list = soup.find("div", attrs={'class':'col'}).ul.find_all('li')
# print(len(paper_list))
#
# exist_papers = []
# if os.path.isfile(output_file_name):
#     with open(output_file_name, 'r') as f:
#         exist_papers = json.load(f)
# print("exist papers", len(exist_papers))
#
# openreview_url = "https://api.openreview.net"
# client = openreview.Client(baseurl=openreview_url)
#
# invitations = set([])
#
# for item in paper_list[len(exist_papers):]:
#     paper = {}
#     paper["link"] = item.a["href"]
#     paper["title"] = item.a.get_text()
#     paper["authors"] = item.i.get_text()
#
#     req = urllib.request.Request(nips_url + paper["link"])
#     resp = urllib.request.urlopen(req)
#     data = resp.read().decode('utf-8')
#     soup = BeautifulSoup(data, 'html.parser')
#     content = soup.find("div", attrs={'class': 'col'})
#
#     hrefs = content.div.find_all("a")
#     for h in hrefs:
#         hcontent = h.get_text()
#
#         # reviews
#         if hcontent == "Reviews And Public Comment »":
#             review_url = h["href"]
#             id = review_url.split("=")[1].strip().split("#")[0]
#             paper["id"] = id
#             notes = client.get_notes(forum=id)
#
#             reviews_commments = []
#             paper_invitations = []
#             time_final_decision = None
#             for note in notes:
#                 # print("cdate",time.localtime(note.cdate/1000))
#                 # print("tcdate", time.localtime(note.tcdate / 1000))
#                 # print("tmdate", time.localtime(note.tmdate / 1000))
#
#                 paper_invitations.append(note.invitation.split("/")[-1])
#                 if "Submission" in note.invitation and note.id == id:
#                     paper["number"] = note.number
#                 elif "Decision" in note.invitation or "Meta_Review" in note.invitation:
#                     # print(note.invitation)
#                     time_final_decision = time.localtime(note.tmdate / 1000)
#                     paper["final_decision"] = note.to_json()
#                 else:
#                     reviews_commments.append(note.to_json())
#             for note in notes:
#                 if note.cdate != None:
#                     ntime = time.localtime(note.cdate / 1000)
#                     if ntime > time_final_decision:
#                         print(time_final_decision, ntime)
#             print("reviews_comments", len(reviews_commments))
#             paper["reviews_commments"] = reviews_commments
#
#             invitation_texts = ",".join(sorted(list(set(paper_invitations))))
#             invitations.add(invitation_texts)
#
#         if hcontent == "Paper":
#             paper["pdf"] = h["href"]
#
#     # abstract
#     paper["abstract"] = content.find_all('p')[-2].get_text()
#     paper["comment"] = "accept"
#     papers.append(paper)
#
#     if len(papers)%50==0:
#         exist_papers = []
#         if os.path.isfile(output_file_name):
#             with open(output_file_name, 'r') as f:
#                 exist_papers = json.load(f)
#         exist_papers.extend(papers)
#         f = open(output_file_name, 'w')
#         f.write(json.dumps(exist_papers))
#         f.close()
#         print("paper count", len(exist_papers))
#         papers = []
#
# print(sorted(invitations))
#
# with open(output_file_name, 'r') as f:
#     exist_papers = json.load(f)
# exist_papers.extend(papers)
# f = open(output_file_name, 'w')
# f.write(json.dumps(exist_papers))
# f.close()
# print("paper count", len(exist_papers))
#
# print("Done")


# The first step is getting the paper list from the OpenReview getting data, and then get reviews with forum ids
import json
import openreview
import time

year = 2021
base_url = "https://api.openreview.net"
client = openreview.Client(baseurl=base_url)
notes = client.get_all_notes(signature='NeurIPS.cc/%s/Conference'%year)# using signature to get all submissions

invitations = set([])
for note in notes:
    invitations.add(note.invitation)

for invitation in invitations:
    print(invitation, len(client.get_all_notes(invitation=invitation)))

papers = []
count = 0
for note in notes:
    paper = {}
    paper["link"] = "https://openreview.net/forum?id=" + note.forum
    content = note.content
    # print(content.keys())
    paper["title"] = content['title']
    paper["authors"] = content['authors']
    paper["abstract"] = content['abstract']
    paper["tl_dr"] = content.get('one-sentence_summary', '')
    paper["keywords"] = content['keywords']

    paper["id"] = note.forum

    notes = client.get_notes(
        forum=paper["id"])  # using forum to get notes of each paper, and notes include the paper information, reviews (official and public) and responses.
    reviews_commments = []
    paper_invitations = []
    time_final_decision = None
    for note in notes:
        # print("cdate",time.localtime(note.cdate/1000))
        # print("tcdate", time.localtime(note.tcdate / 1000))
        # print("tmdate", time.localtime(note.tmdate / 1000))
        paper_invitations.append(note.invitation.split("/")[-1])
        if "Submission" in note.invitation and note.id == id:
            paper["pdf"] = base_url + note.content["pdf"]
            paper["number"] = note.number
        elif "Decision" in note.invitation or "Meta_Review" in note.invitation:
            # print(note.invitation)
            time_final_decision = time.localtime(note.tmdate / 1000)
            paper["final_decision"] = note.to_json()
            paper["comment"] = note.content['decision']
            print(paper['comment'])
        else:
            reviews_commments.append(note.to_json())
    # for note in notes:
    #     if note.cdate != None:
    #         ntime = time.localtime(note.cdate / 1000)
    #         if ntime > time_final_decision:
    #             print(time_final_decision, ntime)
    print("reviews_comments", len(reviews_commments))
    invitation_texts = ",".join(sorted(list(set(paper_invitations))))
    paper["reviews_commments"] = reviews_commments
    if "Blind_Submission" in invitation_texts:
        count += 1
    # print(paper["final_decision"])

    papers.append(paper)

print("blind submission", count)

print(len(papers))
f = open('data/nips_%s.json'%year, 'w')
f.write(json.dumps(papers))
f.close()
