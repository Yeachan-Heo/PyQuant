import pandas as pd
import re
import urllib.request
import urllib.parse
import PyQuant.path as path
from bs4 import BeautifulSoup
import subprocess as sp
import threading
import numpy as np
import time
import os

current_step = 0

total_step = 0

basepath = "~/Documents/Github/PyQuant/data/price/"

time_start = 0

def time_str(sec):
    hr = sec // 3600
    min = (sec - hr*3600) // 60
    sec = sec - hr * 3600 - min * 60
    return (f"{int(hr)}hr {int(min)}min {int(round(sec))}sec")


def get_max_page(url):
    try:
        with urllib.request.urlopen(url) as res:
            p = re.compile("page=\d*")
            html = res.read()
            soup = BeautifulSoup(html, "html.parser")
            a_tags = soup.find_all("a")
            a_tags = list(a_tags)
            last = a_tags[::-1][0]
            last_page = int(p.findall(str(last))[0][5:])
            return last_page
    except:
        return get_max_page(url)

def get_price_data(code):
    try:
        url = f"https://finance.naver.com/item/sise_day.nhn?code={code}"
        max_page = get_max_page(url)
        data = pd.DataFrame()
        for page in (range(1, max_page)):
            df = pd.read_html(url + f"&page={page}")[0]
            data = data.append(df)
        data = data.iloc[::-1]
        data = data.dropna(axis=0)
        data = data.reset_index(drop=True)
        data.columns = ["date","close","diff","open","high","low","volume"]
        return data
    except: return get_price_data(code)

def get_price_data_and_save(path, code):
    code = code[:6]
    path = path + str(code) + ".csv"
    data = get_price_data(str(code))
    data.to_csv(path)

def _main_thread(codes, x):
    for c in codes:
        get_price_data_and_save(path.price_data + x[:-4] + "/", c)
        global current_step
        current_step += 1
        dt = time.time() - time_start
        tps = dt / current_step
        sp.call("clear", shell=True)
        print(f"{current_step}/{total_step} "
              f"{time_str((total_step - current_step)*tps)} left")

def _main(code, x, size=2):
    codes = code[len(os.listdir(path.price_data + x[:-4])):]
    global total_step
    total_step += len(codes)
    for i in (range(0, len(codes), size)):
        f = lambda : _main_thread(codes[i:i+size], x)
        threading.Thread(target=f).start()

def rename():
    for market in os.listdir(path.price_data):
        try: fnames = os.listdir(path.price_data + market)
        except NotADirectoryError: continue
        for fname in fnames:
            fpath = path.price_data + market + "/" + fname
            df = pd.read_csv(fpath)
            print(df)
            df.columns = ["index", "date","close","diff","open","high","low","volume"]
            df.to_csv(fpath)

if __name__ == '__main__':
    time_start = time.time()
    try:
        for x in (os.listdir(path.code_data)):
            code = pd.read_csv(path.code_data + x)["code"]
            threading.Thread(target=lambda : _main(code, x)).start()
    except:
        exit()





