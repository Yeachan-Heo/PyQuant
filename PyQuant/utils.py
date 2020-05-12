import pandas as pd

def read_csv_stock(path, date_column="Date", **kwargs):
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df[date_column].to_list()))
    return df

