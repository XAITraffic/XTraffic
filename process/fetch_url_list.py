#! -*- coding:utf-8 -*-
 
import requests

from io import StringIO
from lxml import etree
import os

district_id = 202


district_id_list = [3, 4,5,6,7,8,10,11,12]
session_id = "24c47b3ff255b9a0b97fc07855b2924f"
session = requests.Session()
session.cookies.set("PHPSESSID", session_id, domain="pems.dot.ca.gov")


def save_file_ex(data, file_name):
    data_dir = 'raw_data/year_2022'
    data_path = os.path.join(data_dir, file_name)
    data_file = open(data_path, "w")
    data_file.write(data)
    data_file.close()


for d in district_id_list:

    site_url = 'https://pems.dot.ca.gov/?srq=clearinghouse&district_id={}&geotag=&yy=2022&type=station_5min&returnformat=text'.format(d)
    
    resp = session.get(site_url)
    print(f"{str(resp.status_code)}")
    resp.raise_for_status()
    data = resp.text
    file_name = 'json_2022_d{}.json'.format(d)
    save_file_ex(data, file_name)
    