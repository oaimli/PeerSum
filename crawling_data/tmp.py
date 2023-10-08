import openreview

base_url = "https://api.openreview.net"
client = openreview.Client(baseurl=base_url)
rcs = client.get_notes(
        forum="x9jS8pX3dkx")
for rc in rcs:
    print(rc)
    print()