import requests, sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep

db = sqlite3.connect("data.db")
db_cursor = db.cursor()
db_cursor.execute("CREATE TABLE IF NOT EXISTS arxiv(id, date, title, abstract, answer, classification)")

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
categories = {"cs.CL"}
token = None

#max_date = next(bigquery_client.query("SELECT MAX(date) AS max_date from llamars.arxiv.arxiv").result(), None).max_date
from_date = str((datetime.now() - timedelta(2)).date()) #str(max_date + timedelta(1) if max_date else (datetime.now() - timedelta(1)).date())

while True:
    url = "https://export.arxiv.org/oai2?verb=ListRecords&"
    if token is None:
        url += "set=cs&metadataPrefix=arXiv&from=" + from_date
    else:
        url += "resumptionToken=" + token.text
    tree = ET.fromstring(requests.get(url).text)
    list_records = tree.find(OAI + "ListRecords")
    for (i, record) in enumerate(list_records.findall(OAI + "record")):
        print(i)
        arxiv = record.find(OAI + "metadata").find(ARXIV + "arXiv")
        paper_categories = set(arxiv.find(ARXIV + "categories").text.split())
        if paper_categories.isdisjoint(categories) or arxiv.find("updated") is not None:
            continue
        id = arxiv.find(ARXIV + "id").text
        date = record.find(OAI + "header").find(OAI + "datestamp").text
        title = " ".join(arxiv.find(ARXIV + "title").text.split())
        abstract = arxiv.find(ARXIV + "abstract").text
        while True:
            try:
                answer = requests.get("http://localhost:5000", params={"prompt": "papers", "input": f"{title}\n{abstract}"}).text
                break
            except:
                print("llm connection error")
                sleep(1)
        classification = "yes" == answer.lower()
        db_cursor.execute("INSERT INTO arxiv VALUES (?, ?, ?, ?, ?, ?)", (id, date, title, abstract, answer, classification))
        db.commit()

    token = list_records.find(OAI+"resumptionToken")
    if token is None or token.text is None:
        break
