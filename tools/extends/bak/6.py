import json
import requests

'''
中金所：
公告api: app.cffex.com.cn/web/service/articles/list
参数：params=[0,2]
公告详情api：https://app.cffex.com.cn/web/service/articles/detail
参数：params=[10420]
'''

'''
郑州商品交易所
公告api:http://zceapp.epolestar.xyz/api/noticeMsg
参数：params={"page": 1, "per_page": 10}
公告详情api：http://zceapp.epolestar.xyz/api/noticeMsg/Id
'''

HOST = "http://zceapp.epolestar.xyz"
search_data = {"page": 1, "per_page": 10}
notice_list = requests.post(HOST + "/api/noticeMsg", data=json.dumps(search_data))
if notice_list.status_code != 200:
    print("请求错误！")
jsonObj = json.loads(notice_list.text)
for row in jsonObj['data']:
    Id = row['Id']
    http_res = requests.post(HOST + "/api/noticeMsg/{}".format(str(Id)))
    notice = json.loads(http_res.text)
    print(notice)
print(jsonObj)
