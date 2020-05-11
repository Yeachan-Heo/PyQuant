import modin.pandas as pd
import yfinance as yf
import numpy as np
import datetime

from typing import *

class BackTestCore:
    def __init__(self, seed_money):
        self.idx = 0
        self.seed_money = seed_money
        self.money = seed_money
        self.locked = False
        self.df = pd.DataFrame()
        self.df_index = []
        self.data_dict = {}
        self.portfolio = {}
        self.datetime_idx = None

    def __getitem__(self, item):
        return self.df[item]

    @property
    def available_stocks(self):
        return list(self.df.columns)

    def max_amount(self, code):
        if not code in self.df.columns: raise ValueError(f"Invalid code:{code}")
        current_price = self.df[code].iloc[self.df.index.index(self.datetime_idx)]
        return self.money // current_price

    def move_index(self, code):

    def register_data_dataframe(self, code, df:pd.DataFrame, high_column="High", low_column="Low"):
        if len(df.index) > self.df_index:
            self.df_index = df.index
        self.data_dict[code] = ((df[high_column] + df[low_column])/2).to_list()

    def register_data_code(self, code:str, **history_args):
        ticker = yf.Ticker(code)
        hist = ticker.history(**(history_args if history_args else {"period":"max"}))
        self.register_data_dataframe(code, hist)

    def register_data_path(self, code, path, **kwargs):
        df = pd.read_csv(path)
        self.register_data_dataframe(code, df, **kwargs)

    def lock(self, min=None, max=None):
        length_lst = np.array(list(map(len, self.data_dict.values())))
        nan_sizes = np.max(length_lst) - length_lst
        for nan_size, (key, value) in zip(nan_sizes, self.data_dict.items()):
            self.data_dict[key] = [np.nan]*nan_size + value
        self.df = pd.DataFrame(self.data_dict, index=pd.DatetimeIndex(pd.to_datetime(self.df_index)))
        del self.data_dict
        if min:
            max = datetime.datetime.now().date() if not max else max
            self.df = self.df.loc[min<=self.df.index<=max]
        self.datetime_idx = self.df.index[0]
        self.locked = True

    def islocked(self):
        assert self.locked, "you have to lock this engine by calling engine.lock() before using"

    def reset(self):
        self.portfolio.clear()
        self.money = self.seed_money






