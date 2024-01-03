import os
from tweepy import OAuth2UserHandler
from dotenv import load_dotenv

load_dotenv()

oauth2_user_handler = OAuth2UserHandler(
    client_id = os.environ["CLIENT_ID"],
    redirect_uri = "https://localhost",
    scope = ["tweet.read", "users.read", "tweet.write", "offline.access"],
    client_secret = os.environ["CLIENT_SECRET"]
)

print(oauth2_user_handler.get_authorization_url())
authorization_response = input()
access_token = oauth2_user_handler.fetch_token(authorization_response)
print(access_token)
