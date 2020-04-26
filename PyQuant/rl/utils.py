import PyQuant.path as path
import pandas as pd
import numpy as np
import threading
import os

def thread(func):
    def wrapped(*args, **kwargs):
        threading.Thread(
            target=lambda:func(*args, **kwargs)).start()
    return wrapped

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def get_random_data():
    market = np.random.choice(os.listdir(path.price_data))
    try: fname = np.random.choice(os.listdir(path.price_data + market))
    except: return get_random_data()
    fpath = path.price_data + market + "/" + fname
    return pd.read_csv(fpath)

