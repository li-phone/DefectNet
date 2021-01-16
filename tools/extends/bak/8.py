import json
from pandas.io.json import json_normalize
import requests
from bs4 import BeautifulSoup

HOST = "http://app.czce.com.cn"
TABLE_API = "/cms/pub/search/searchdt.jsp"

search_data = {
    "DtName": "保证金",
    "DtbeginDate": "2011-01-01",
    "DtendDate": "2021-01-15",
    "__go2pageNum": 1
}

result = requests.post(HOST + TABLE_API, data=search_data)
if result.status_code != 200:
    print("请求错误！")
soup = BeautifulSoup(result.text, "html.parser")
tbody = soup.find_all(name="tr")
save_results = []
for tr in tbody:
    td = tr.find_all("td")
    if len(td) == 3:
        href = td[1].a.get("href")
        title = td[1].a.text
        date = td[2].text
        save_results.append(dict(title=title, href=href, date=date))
save_results = json_normalize(save_results)
print(save_results)
