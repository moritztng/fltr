import urllib, urllib.request, requests
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
        url += f"from={(datetime.now() - timedelta(2)).date()}&set=cs&metadataPrefix=arXiv"
    else:
        url += "resumptionToken=" + token.text
    data = urllib.request.urlopen(url)
    tree = ET.fromstring(data.read().decode('utf-8'))
    for (i, record) in enumerate(tree.find(OAI + "ListRecords").findall(OAI + "record")):
        arxiv = record.find(OAI + "metadata").find(ARXIV + "arXiv")
        id = arxiv.find(ARXIV + "id").text
        title = arxiv.find(ARXIV + "title").text
        output = requests.get("http://localhost:5000", params={"prompt": "ml", "input": title}).text
        bigquery_client.insert_rows(bigquery_client.get_table("llamars.arxiv.arxiv"), [{"id": id, "title": title, "output": output}])
        print(i)
    token = tree.find(OAI + 'ListRecords').find(OAI+"resumptionToken")
    if token is None or token.text is None:
        break
    sleep(5)
