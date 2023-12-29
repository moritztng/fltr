import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep
from google.cloud import bigquery

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
token = None
bigquery_client = bigquery.Client()

while True:
    url = "https://export.arxiv.org/oai2?verb=ListRecords&"
    if token is None:
        url += "set=cs&metadataPrefix=arXiv&from=" + str((datetime.now() - timedelta(3)).date())
    else:
        url += "resumptionToken=" + token.text
    tree = ET.fromstring(requests.get(url).text)
    list_records = tree.find(OAI + "ListRecords")
    for (i, record) in enumerate(list_records.findall(OAI + "record")):
        arxiv = record.find(OAI + "metadata").find(ARXIV + "arXiv")
        id = arxiv.find(ARXIV + "id").text
        title = arxiv.find(ARXIV + "title").text
        output = None
        while output is None:
            try:
                output = requests.get("http://localhost:5000", params={"prompt": "topic", "input": title}).text
            except:
                print("llm connection error")
                sleep(1)
        bigquery_client.insert_rows(bigquery_client.get_table("llamars.arxiv.arxiv"), [{"id": id, "title": title, "output": output}])
        print(i)
    token = list_records.find(OAI+"resumptionToken")
    if token is None or token.text is None:
        break
