import os, requests, sqlite3
from base64 import b64encode
from dotenv import load_dotenv

db = sqlite3.connect("data.db")
db_cursor = db.cursor()
db_cursor.execute("CREATE TABLE IF NOT EXISTS posts(id)")

load_dotenv()
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]

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

rows = db_cursor.execute("SELECT id, title FROM arxiv LEFT JOIN posts USING(id) WHERE posts.id IS NULL AND classification = true")
for (id, title) in rows:
    response = requests.request(
        "POST",
        "https://api.twitter.com/2/tweets",
        json = {
            "text": f"{title}\nhttps://arxiv.org/abs/{id}"
        },
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
    )
    db_cursor.execute("INSERT INTO posts VALUES (?)", (id,))
    db.commit()
    print((id, title), response.text)
