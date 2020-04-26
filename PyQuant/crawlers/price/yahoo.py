
import PyQuant.path as path
import yfinance as yf
import pandas as pd
import tqdm
import os

def get_data(code):
    ticker = yf.Ticker(code)
    df = ticker.history(period='max')
    return df

def get_price_data_and_save(path, code):
    df = get_data(code)
    df.to_csv(path+code+".csv")
