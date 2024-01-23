import asyncio, os, requests, sqlite3, subprocess
import xml.etree.ElementTree as ET
from typing import Callable
from datetime import datetime, timedelta
from base64 import b64encode
from dotenv import load_dotenv

db = sqlite3.connect("data.db")


async def fetch_arxiv():
    db_cursor = db.cursor()
    db_cursor.execute(
        "CREATE TABLE IF NOT EXISTS arxiv(id, date, title, abstract, answer, classification)"
    )

    OAI = "{http://www.openarchives.org/OAI/2.0/}"
    ARXIV = "{http://arxiv.org/OAI/arXiv/}"
    categories = {"cs.CL"}
    token = None

    # max_date = next(bigquery_client.query("SELECT MAX(date) AS max_date from llamars.arxiv.arxiv").result(), None).max_date
    from_date = str(
        (datetime.now() - timedelta(3)).date()
    )  # str(max_date + timedelta(1) if max_date else (datetime.now() - timedelta(1)).date())

    while True:
        url = "https://export.arxiv.org/oai2?verb=ListRecords&"
        if token is None:
            url += "set=cs&metadataPrefix=arXiv&from=" + from_date
        else:
            url += "resumptionToken=" + token.text
        tree = ET.fromstring(requests.get(url).text)
        list_records = tree.find(OAI + "ListRecords")
        for i, record in enumerate(list_records.findall(OAI + "record")):
            print(i)
            arxiv = record.find(OAI + "metadata").find(ARXIV + "arXiv")
            paper_categories = set(arxiv.find(ARXIV + "categories").text.split())
            if (
                paper_categories.isdisjoint(categories)
                or arxiv.find("updated") is not None
            ):
                continue
            id = arxiv.find(ARXIV + "id").text
            date = record.find(OAI + "header").find(OAI + "datestamp").text
            title = " ".join(arxiv.find(ARXIV + "title").text.split())
            abstract = arxiv.find(ARXIV + "abstract").text
            while True:
                try:
                    answer = requests.get(
                        "http://localhost:5000",
                        params={"prompt": "papers", "input": f"{title}\n{abstract}"},
                    ).text
                    break
                except:
                    print("llm connection error")
                    await asyncio.sleep(1)
            classification = "yes" == answer.lower()
            db_cursor.execute(
                "INSERT INTO arxiv VALUES (?, ?, ?, ?, ?, ?)",
                (id, date, title, abstract, answer, classification),
            )
            db.commit()

        token = list_records.find(OAI + "resumptionToken")
        if token is None or token.text is None:
            break


async def post_x():
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
                "client_id": client_id,
            },
            headers={
                "Authorization": f"Basic {b64encode(f'{client_id}:{client_secret}'.encode()).decode()}",
            },
        ).json()
        f.seek(0)
        f.write(response["refresh_token"])
        access_token = response["access_token"]

    rows = db.cursor().execute(
        "SELECT id, title FROM arxiv LEFT JOIN posts USING(id) WHERE posts.id IS NULL AND classification = true"
    )
    for id, title in rows:
        response = requests.request(
            "POST",
            "https://api.twitter.com/2/tweets",
            json={"text": f"{title}\nhttps://arxiv.org/abs/{id}"},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        db_cursor.execute("INSERT INTO posts VALUES (?)", (id,))
        db.commit()
        print((id, title), response.text)


async def schedule(start: datetime, interval: timedelta, task: Callable[[None], None]):
    while True:
        delay = int((start - datetime.now()).total_seconds())
        await asyncio.sleep(
            delay if delay >= 0 else delay % int(interval.total_seconds())
        )
        await task()


async def main():
    subprocess.Popen(["cargo", "run", "--release", "server"])

    fetch_arxiv_task = asyncio.create_task(
        schedule(datetime(2024, 1, 23, 3, 53, 00), timedelta(days=1), fetch_arxiv)
    )
    post_x_task = asyncio.create_task(
        schedule(datetime(2024, 1, 25, 1, 26, 00), timedelta(days=1), post_x)
    )

    await fetch_arxiv_task
    await post_x_task


asyncio.run(main())
