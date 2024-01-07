import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
categories = {"cs.CL"}
token = None
bigquery_client = bigquery.Client()

max_date = next(bigquery_client.query("SELECT MAX(date) AS max_date from llamars.arxiv.arxiv").result(), None).max_date
from_date = str(max_date + timedelta(1) if max_date else (datetime.now() - timedelta(1)).date())

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
        title = arxiv.find(ARXIV + "title").text
        abstract = arxiv.find(ARXIV + "abstract").text
        while True:
            try:
                output = requests.get("http://localhost:5000", params={"prompt": "papers", "input": f"{title}\n{abstract}"}).text
                break
            except:
                print("llm connection error")
                sleep(1)
        output = output.split("Yes or No")
        explanation, answer = output if len(output) == 2 else (output[0], "")
        classification = "yes" in answer.lower()
        errors = bigquery_client.insert_rows(bigquery_client.get_table("llamars.arxiv.arxiv"), [{"id": id, "date": date, "title": title, "abstract": abstract, "explanation": explanation, "answer": answer, "classification": classification}])
        if errors:
            raise ValueError(errors)
        
    token = list_records.find(OAI+"resumptionToken")
    if token is None or token.text is None:
        break
