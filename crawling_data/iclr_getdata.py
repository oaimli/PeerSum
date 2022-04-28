# The first step is getting the paper list from the OpenReview getting data, and then get reviews with forum ids
# Data for ICLR 2017 is from PeerRead (https://github.com/allenai/PeerRead), from 2018 data can be obtained from OpenReview
import json
import openreview
import time

year = 2022
base_url = "https://api.openreview.net"
client = openreview.Client(baseurl=base_url)
notes = client.get_all_notes(signature='ICLR.cc/%s/Conference'%year)# using signature to get all submissions

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


print(len(papers))
f = open('data/iclr_%s.json'%year, 'w')
f.write(json.dumps(papers))
f.close()
