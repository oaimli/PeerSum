# The first step is getting the paper list from the OpenReview getting data, and then get reviews with forum ids
import json
import openreview
import time

year = 2022
base_url = "https://api.openreview.net"
client = openreview.Client(baseurl=base_url)
notes = client.get_all_notes(signature='NeurIPS.cc/%s/Conference'%year)# using signature to get all submissions
print("all papers", len(notes))

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
    paper["title"] = content['title']
    paper["authors"] = content['authors']
    paper["abstract"] = content['abstract']
    paper["tl_dr"] = content.get('TL;DR', "")
    paper["keywords"] = content['keywords']
    paper["id"] = note.forum

    rcs = client.get_notes(
        forum=paper["id"])  # using forum to get notes of each paper, and notes include the paper information, reviews (official and public) and responses.
    reviews_commments = []
    paper_invitations = []
    time_final_decision = None
    decision = ""
    recommendation = ""
    for rc in rcs:
        # print(note.invitation)
        # print("cdate",time.localtime(note.cdate/1000))
        # print("tcdate", time.localtime(note.tcdate / 1000))
        # print("tmdate", time.localtime(note.tmdate / 1000))
        paper_invitations.append(rc.invitation.split("/")[-1])
        if "Submission" in rc.invitation and note.id == paper["id"]:
            paper["pdf"] = base_url + rc.content["pdf"]
            paper["number"] = rc.number
        elif "Meta_Review" in rc.invitation:
            # print(note.invitation)
            time_final_decision = time.localtime(rc.tmdate / 1000)
            paper["final_decision"] = rc.to_json()
            print(rc.content['recommendation'])
            recommendation = rc.content['recommendation']
            # print(paper['comment'])
        elif "Decision" in rc.invitation:
            print(rc.content['decision'])
            paper["comment"] = rc.content['decision']
            decision = rc.content['decision']
        else:
            reviews_commments.append(rc.to_json())
    # for note in notes:
    #     if note.cdate != None:
    #         ntime = time.localtime(note.cdate / 1000)
    #         if ntime > time_final_decision:
    #             print(time_final_decision, ntime)
    if recommendation != decision:
        print(paper["link"])
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
