import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import random
import pprint
import json
import time
from typing import *
from numba import jit
from PyQuant import path, utils
from dateutil.relativedelta import relativedelta
random.seed(100)
def with_seed(seed=100):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            random.seed(seed)
            return func(*args, **kwargs)
        return wrapped
    return wrapper

random.randint = with_seed()(random.randint)
random.choice = with_seed()(random.choice)


def NamedTimer(name):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            s = time.time()
            ret = func(*args, **kwargs)
            print(f"{name}:",time.time()-s)
            return ret
        return wrapped
    return wrapper

pp = pprint.PrettyPrinter(indent=4)

class StockBase:
    def __init__(self, df):
        self.df = df

    @property
    def index(self):
        return self.df.index

    def __getitem__(self, item):
        idx = list(self.df.index).index(item)
        return self.df["Mid"].iloc[idx]

class BackTestCore:
    def __init__(self, seed_money, print_log=True, fee=0.00015, tax=0.003):
        self.seed_money = seed_money
        self.money = seed_money
        self.locked = False
        self.stock_data = {}
        self.stock_data_index = []
        self.data_dict = {}
        self.portfolio = {}
        self.datetime_idx = None
        self.print_log = print_log
        self.fee, self.tax = fee, tax

    def __getitem__(self, item):
        return self.stock_data[item]
    #
    def pv_log(self):
        print(f"[{str(self.datetime_idx)[:10]}] pv:{self.pv}, yield:{(self.pv/self.seed_money-1)*100}%\n")

    @property
    #@NamedTimer("buyablestocks")
    def buyable_stocks(self):
        return list(filter(
            lambda x: self.stock_data[x].index[0] <= self.datetime_idx,
            self.stock_data.keys()
        ))

    @property
    def sellable_stocks(self):
        return list(filter(
            lambda x: self.portfolio[x]["amount"] > 0,
            self.portfolio.keys()
        ))

    @property
    def idx(self):
        return self.get_past_idx()

    @staticmethod
    @jit
    def get_stock_value(amounts, prices):
        val = 0
        for amount, price in zip(amounts, prices):
            val += amount*price

    @property
    #@NamedTimer("pv")
    def pv(self):
        stock_value = np.sum(list(map(lambda x: self.portfolio[x]["amount"]*self.current_price(x), self.portfolio.keys())))
        #stock_value = self.get_stock_value(self.portfolio, map(self.current_price, self.portfolio.keys()))
        return stock_value + self.money

    def get_past_idx(self, y=0, m=0, d=0):
        try:
            return list(self.stock_data_index).index(self.datetime_idx - relativedelta(years=y, months=m, days=d))
        except ValueError:
            return self.get_past_idx(y, m, d+1)

    #@NamedTimer("maxamount")
    def max_amount(self, code):
        if not code in self.stock_data.keys(): raise ValueError(f"Invalid code:{code}")
        current_price = self.current_price(code)
        amount = self.money // current_price
        while self.money - current_price*self.fee*amount < 0:
            amount -= 1
            if amount <= 0: return 0
        return amount

    def price_at(self, code, date):
        try:
            price = self.stock_data[code][date].squeeze()
            if price <= 0:
                print(price, code, date)
            return price
        except Exception as E:
            return self.stock_data[code].df.loc[self.stock_data[code].index < date]["Mid"].iloc[-1]

    def current_price(self, code):
        return self.price_at(code, self.datetime_idx)

    def move_index(self, y=0, m=0, d=0):
        assert y>=0 & m>=0 & d>=0, "can't go backward"
        self.datetime_idx += relativedelta(years=y, months=m, days=d)
        self.update_yield()
        if self.datetime_idx >= self.stock_data_index[-1]:
            return True
        if not self.datetime_idx in self.stock_data_index:
            return self.move_index(d=1)
        return False

    def update_yield(self):
        for key, value in self.portfolio.items():
            cprice = self.current_price(key)
            self.portfolio[key]["gain"] = (cprice - value["avg_price"])*value["amount"]
            self.portfolio[key]["yield(%)"] = (cprice/value["avg_price"]-1)*100

    #@NamedTimer("stockio")
    def _stock_io(self, code, cprice, amount):
        self.money -= amount * cprice
        if not amount == 0:
            self.portfolio[code]["amount"] += amount

    #
    def sell(self, code, amount):
        if not code in self.portfolio.keys():
            print(f"unsellable stock :{code}")
            return

        cprice = self.current_price(code)
        amount = min(self.portfolio[code]["amount"], amount)

        self._stock_io(code, cprice, -amount)

        tax = self.tax * cprice * amount
        fee = (amount * cprice * self.fee)
        self.money -= (tax+fee)

        if amount > 0:
            realized_gain = amount * (cprice - self.portfolio[code]["avg_price"])-tax-fee
            self.portfolio[code]["realized_gain"] += realized_gain
            self.portfolio[code]["history"][str(self.datetime_idx)] = {
                "amount": -amount,
                "price": cprice,
                "realized_gain" : realized_gain,
                "realized_yield(%)" : (realized_gain/amount/self.portfolio[code]["avg_price"])*100
            }

        if self.print_log:
            print(f"sell {code} {amount}")
            self.pv_log()


    #
    def buy(self, code, amount):
        assert amount >= 0
        amount = min(amount, self.max_amount(code))
        cprice = self.current_price(code)
        if ((not code in self.portfolio.keys()) & (amount > 0)):
            self.portfolio[code] = {"amount": 0, "avg_price": 0, "history": {}, "realized_gain":0}

        fee = amount * cprice * self.fee
        self.money -= fee

        if amount > 0:
            self.portfolio[code]["avg_price"] = \
                (
                        self.portfolio[code]["avg_price"] * self.portfolio[code]["amount"]
                        + cprice * amount
                ) / (self.portfolio[code]["amount"] + amount)

            self.portfolio[code]["history"][str(self.datetime_idx)] = {
                "amount": amount,
                "price": cprice
            }

        self._stock_io(code, cprice, amount)

        if self.print_log:
            print(f"buy {code} {amount}")
            self.pv_log()

    def register_data_dataframe(self, code, df:pd.DataFrame, high_column="High", low_column="Low", date_column="Date"):
        df["Mid"] = ((df[high_column] + df[low_column])/2).to_list()
        df.index = pd.DatetimeIndex(df[date_column])
        df = df.fillna(method="bfill")
        df = df.loc[df["Low"] > 0]
        self.stock_data[code] = StockBase(df)
        if self.stock_data_index.__len__() < len(df.index):
            self.stock_data_index = df.index
        print(f"registering:{code}, {df.index[0]}~{df.index[-1]}")

    def register_data_code(self, code:str, **history_args):
        ticker = yf.Ticker(code)
        hist = ticker.history(**(history_args if history_args else {"period":"max"}))
        self.register_data_dataframe(code, hist)

    def register_data_path(self, code, path, **kwargs):
        df = utils.read_csv_stock(path, **kwargs)
        self.register_data_dataframe(code, df, **kwargs)

    def lock(self):
        self.datetime_idx = self.stock_data_index[0]
        self.locked = True

    def islocked(self):
        assert self.locked, "you have to lock this engine by calling engine.lock() before using"

    def reset(self):
        self.portfolio.clear()
        self.money = self.seed_money

import os

if __name__ == '__main__':
    seed_money = 100000000
    codes = pd.read_csv(path.code_data + "kospi.csv")["code"].to_list()
    engine = BackTestCore(seed_money, print_log=False)
    for p in os.listdir(path.kospi_price_data):
        engine.register_data_path(p[:-4], path.kospi_price_data + p)
    engine.lock()
    engine.move_index(y=5)
    i = 0
    break_flag = False
    while not break_flag:
        randombuy = random.choice(engine.buyable_stocks)
        engine.buy(randombuy, random.randint(0, engine.max_amount(randombuy)+1))
        if engine.move_index(d=14): break_flag = True
        if engine.sellable_stocks:
            randomsell = random.choice(engine.sellable_stocks)
            engine.sell(randomsell, random.randint(0, engine.portfolio[randomsell]["amount"]+1))
        if i % 100 == 0:
            pp.pprint(engine.portfolio)
            engine.pv_log()
        i += 1
    pp.pprint(engine.portfolio)
    json.dump(engine.portfolio, open("portfolio.json", "w"))
    print(engine.pv / seed_money * 100-100, "% 수익!")





