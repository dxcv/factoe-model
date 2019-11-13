import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from toy_backtest_v2 import get_stock_close_price
from data_process import read_csv, get_name2code


def get_ts_code_original_weight(hs300_wgt_data, date=None):
    if date:
        hs300_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == date]
    original_weight = {}
    num = len(hs300_wgt_data)
    date_list = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
    assert len(date_list) == 1, print('the {} hs300_wgt_data has more one days data'.format(date_list))
    assert num >= 280, print('the num of hs300 is not 300 in date {} '.format(date_list), 'actual num is {}'.format(num))
    for i in range(num):
        tmp = hs300_wgt_data.iloc[i]
        ts_code = tmp['ts_code']
        weight = tmp['weight']
        assert ts_code is not None, print('check the ts_code in date {} code is {}'.format(date, ts_code))
        assert weight is not None, print('check the weight in date {} code is {}'.format(date, weight))
        original_weight[ts_code] = weight/100
    assert abs(1 - sum(original_weight.values())) <= 0.01, print(sum(original_weight.values()))
    return original_weight


def select_suspend_stock(trade_code, wgt_code):
    trade_code = set(trade_code)
    wgt_code = set(wgt_code)
    suspend_stock = wgt_code.difference(trade_code)
    return suspend_stock


def get_market_value(trade_data):
    market_val = {}
    num = len(trade_data)
    for i in range(num):
        tmp = trade_data.iloc[i]
        ts_code = tmp['ts_code']
        circ_mv = tmp['circ_mv']
        total_mv = tmp['total_mv']
        weight = circ_mv/total_mv
        weight = myround(weight)
        market_val[ts_code] = weight * total_mv
    return market_val


def caculate_weight_by_market_value(wgt_code, market_value):
    total_mv = 0
    wgt = {}
    for code in wgt_code:
        total_mv += market_value[code]
        wgt[code] = market_value[code]
    for key, val in wgt.items():
        wgt[key] = val/total_mv
    return wgt


def compute_market_value_by_shares(shares, price):
    mv = {}
    for k, v in shares.items():
        mv[k] = shares[k] * price[k]
    return mv


def get_share(trade_data, share_name):
    share = {}
    for i in range(len(trade_data)):
        tmp = trade_data.iloc[i]
        sh = tmp[share_name]
        ts_code = tmp['ts_code']
        share[ts_code] = sh
    return share


def myround(x):
    x = np.piecewise(
        x,
        [x <= 0.15, 0.15 < x <= 0.2, 0.2 < x <= 0.3, 0.3 < x <= 0.4, 0.4 < x <= 0.5,
         0.5 < x <= 0.6, 0.6 < x <= 0.7, 0.7 < x <= 0.8, x > 0.8],
        [lambda x:math.ceil(x*100)/100, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    )
    return x


def describe_diff(a, b, wgt_df_day, code2name):
    wgt_code = wgt_df_day['ts_code'].drop_duplicates().to_list()
    diff = {}
    for code in wgt_code:
        diff[code] = np.abs(a[code] - b[code])
    val2key = dict([(v, k) for k, v in diff.items()])
    print('min stock', code2name[val2key[min(diff.values())]])
    print('max stock', code2name[val2key[max(diff.values())]])
    tmp = pd.Series(list(diff.values()))
    print(tmp.describe(percentiles=np.array(range(1, 101))/100))


def main():
    wgt_path = r'./data/HS300_idx_wt.csv'
    trade_data_path = r'./data/tmp/fixed_daily_data.csv'
    idx_df = r'./data/HS300_idx_wt.csv'
    name2code = get_name2code(wgt_path)
    code2name = dict([(v, k) for k, v in name2code.items()])
    wgt_df = read_csv(wgt_path)
    trade_df = read_csv(trade_data_path)
    idx_df = read_csv(idx_df)
    date_list = wgt_df['trade_date'].drop_duplicates().to_list()
    date_list.sort()
    for date in date_list[::-1]:
        date = 20190531
        idx_df_day = idx_df[idx_df.trade_date == date]
        wgt_df_day = wgt_df[wgt_df.trade_date == date]
        trade_df_day = trade_df[trade_df.trade_date == date][['ts_code', 'trade_date', 'total_mv',
                                                              'circ_mv', 'total_share', 'close',
                                                              'float_share', 'free_share']]
        wgt_code = idx_df_day['ts_code'].drop_duplicates().to_list()
        trade_code = trade_df_day['ts_code'].drop_duplicates().to_list()
        suspended_stock = select_suspend_stock(trade_code, wgt_code)
        if len(suspended_stock) == 0:
            print(date)
            original_weight = get_ts_code_original_weight(idx_df_day)
            mkt_val = get_market_value(trade_df_day)
            wgt = caculate_weight_by_market_value(wgt_code, mkt_val)
            for i in range(len(trade_code)-1):
                a, b = trade_code[i], trade_code[i+1]
                try:
                    r_a = original_weight[a]
                    r_b = original_weight[b]
                    b1 = r_a/r_b
                    w_a = wgt[a]
                    w_b = wgt[b]
                    b2 = w_a/w_b
                    print(b1-b2)
                except:
                    continue
            exit()


if __name__ == '__main__':
    main()