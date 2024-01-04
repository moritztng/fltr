import os, requests
from base64 import b64encode
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]

bigquery_client = bigquery.Client()

with open("refresh_token", "r+") as f:
    refresh_token = f.read()
    response = requests.request(
        "POST",
        "https://api.twitter.com/2/oauth2/token",
        data={
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
            "client_id": client_id,},
        headers={
            "Authorization": f"Basic {b64encode(f'{client_id}:{client_secret}'.encode()).decode()}",
        },
    ).json()
    f.seek(0)
    f.write(response["refresh_token"])
    access_token = response["access_token"]

rows = bigquery_client.query("SELECT id, title FROM llamars.arxiv.arxiv LEFT JOIN llamars.arxiv.posts posts USING(id) WHERE posts.id IS NULL AND classification = true").result()

for row in rows:
    response = requests.request(
        "POST",
        "https://api.twitter.com/2/tweets",
        json = {
            "text": f"{row.title}\nhttps://arxiv.org/pdf/{row.id}.pdf"
        },
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
    )
    bigquery_client.insert_rows(bigquery_client.get_table("llamars.arxiv.posts"), [{"id": row.id}])
    print(row, response.text)
