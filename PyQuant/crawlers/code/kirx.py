import pandas as pd
import PyQuant.path as path
import os
import urllib.parse
import pandas as pd

MARKET_CODE_DICT = {
    'kospi': 'stockMkt',
    'kosdaq': 'kosdaqMkt',
    'konex': 'konexMkt'
}

MARKET_SEP_DICT = {
    "kospi":".KS",
    "kosdaq":".KQ",
    "konex":".KN"
}

DOWNLOAD_URL = 'kind.krx.co.kr/corpgeneral/corpList.do'

def download_stock_codes(market=None, delisted=False):
    params = {'method': 'download'}

    if market.lower() in MARKET_CODE_DICT:
        params['marketType'] = MARKET_CODE_DICT[market]

    if not delisted:
        params['searchType'] = 13

    params_string = urllib.parse.urlencode(params)
    request_url = urllib.parse.urlunsplit(['http', DOWNLOAD_URL, '', params_string, ''])

    df = pd.read_html(request_url, header=0)[0]
    df.종목코드 = df.종목코드.map('{:06d}'.format)
    df.종목코드 = list(map(lambda x: x + MARKET_SEP_DICT[market], df.종목코드))
    df.rename({"종목코드":"code", "회사명":"name"}, inplace=True)
    df = df[["회사명", "종목코드"]]
    df = df.rename(columns={"회사명":"name", "종목코드":"code"})
    df.to_csv(path.code_data + market + ".csv")
    return df

if __name__ == '__main__':
    download_stock_codes("kosdaq")
    download_stock_codes("kospi")
    download_stock_codes("konex")
    
