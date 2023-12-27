import urllib, urllib.request
import xml.etree.ElementTree as ET

OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"

url = 'https://export.arxiv.org/oai2?verb=ListRecords&from=2023-12-22&metadataPrefix=arXiv'
data = urllib.request.urlopen(url)
tree = ET.fromstring(data.read().decode('utf-8'))
for record in tree.find(OAI + "ListRecords").findall(OAI + "record"):
    print(record.find(OAI + "metadata").find(ARXIV + "arXiv").find(ARXIV + "abstract").text)
