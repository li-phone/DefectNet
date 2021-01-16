from selenium import webdriver
import numpy as np

from pandas.io.json import json_normalize
from time import sleep
from selenium import webdriver

# 用webdriver启动浏览器
driver = webdriver.Chrome(executable_path="D:/home/chromedriver.exe")

driver.get("http://www.dce.com.cn/dalianshangpin/yw/fw/jystz/ywtz/index.html")

ul = driver.find_element_by_class_name("list_tpye06")
lis = ul.find_elements_by_tag_name("li")
save_results = []
for li in lis:
    date = li.find_elements_by_tag_name("span")[0].text
    href = li.find_elements_by_tag_name("a")[0].get_attribute(name="href")
    title = li.find_elements_by_tag_name("a")[0].get_attribute(name="title")
    save_results.append(dict(title=title, href=href, date=date))
save_results = json_normalize(save_results)
print(save_results)
