import os, requests
from base64 import b64encode
from dotenv import load_dotenv

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

response = requests.request(
    "POST",
    "https://api.twitter.com/2/tweets",
    json={"text": "hi"},
    headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    },
)

print(response.text)
