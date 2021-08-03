
import re
from io import BytesIO, StringIO
import requests
import pandas as pd

YEAR: int = 2017
url = f"https://www.census.gov/naics/?58967?yearbck={YEAR}"
url_11 = "https://www.census.gov/naics/resources/model/dataHandler.php?input=11&chart=2017&search=Y"

descriptions_url = "https://www.census.gov/naics/resources/js/data/naics-sector-descriptions.json"

with requests.Session() as sess:
    resp = sess.get(url)
    if resp.status_code == 200:
        # print(resp.headers)
        bdata = resp.content
        data = resp.text


btn_data_code_ptrn = r'data-code.+"(\d+)"'
p = re.compile(btn_data_code_ptrn, flags = re.I | re.M)
p.findall(data)
data[:500]


def chunkprint(obj, stepsize = 500):
    start = 0
    stop = stepsize
    obj_len = len(obj)
    while True:
        try:
            yield obj[start:stop]
            start = stop
            stop = start + stepsize
            print(f"Chunks {start} - {stop-1} out of {obj_len}\t{(start/obj_len)*100:.2f}% complete\n")

        except StopIteration:
            return


cp = chunkprint(bdata, 1000)
print(next(cp))

for line in re.split(rb">\n?\s+<", bdata):
    if b"button" in line:
        print(line)