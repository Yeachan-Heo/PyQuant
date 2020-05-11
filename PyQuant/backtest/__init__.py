import modin.pandas as pd
import yfinance as yf
import numpy as np

from typing import *

class BackTestCore:
    def __init__(self, seed_money=1000, period:Tuple[str, str]):
        self.idx = 0
        self.seed_money = seed_money
        self.money = seed_money
        self.locked = False
        self.df = pd.DataFrame()
        self.data_dict = {}
        self.portfolio = {}

    def register_data_dataframe(self, code, df:pd.DataFrame, high_column="High", low_column="Low"):
        self.data_dict[code] = ((df[high_column] + df[low_column])/2).to_list()

    def register_data_code(self, code:str, **history_args):
        ticker = yf.Ticker(code)
        hist = ticker.history(**(history_args if history_args else {"period":"max"}))
        self.register_data_dataframe(code, hist)

    def register_data_path(self, code, path, **kwargs):
        df = pd.read_csv(path)
        self.register_data_dataframe(code, df, **kwargs)

    def lock(self):
        length_lst = np.array(list(map(len, self.data_dict.values())))
        nan_sizes = np.max(length_lst) - length_lst
        for nan_size, (key, value) in zip(nan_sizes, self.data_dict.items()):
            self.data_dict[key] = [np.nan]*nan_size + value
        self.df = pd.DataFrame(self.data_dict)
        del self.data_dict
        self.locked = True

    def islocked(self):
        assert self.locked, "you have to lock this engine by calling engine.lock() before using"

    def buy(self, ):


