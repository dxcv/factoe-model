import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
import os


def myround(x):
    conds = [x <= 0.15,
             (x > 0.15) & (x <= 0.2),
             (x > 0.2) & (x <= 0.3),
             (x > 0.3) & (x <= 0.4),
             (x > 0.4) & (x <= 0.5),
             (x > 0.5) & (x <= 0.6),
             (x > 0.6) & (x <= 0.7),
             (x > 0.7) & (x <= 0.8),
             x > 0.8]
    funcs = [lambda y: np.ceil(y * 100)/100,
             lambda y: 0.2,
             lambda y: 0.3,
             lambda y: 0.4,
             lambda y: 0.5,
             lambda y: 0.6,
             lambda y: 0.7,
             lambda y: 0.8,
             lambda y: 1.0]
    x = np.piecewise(x,conds,funcs)
    return x

def win(x, trim=0.2, rm=True):
    y = x.copy()
    x.dropna()
    if (trim < 0) | (trim > 0.5):
        print("trimming must be reasonable")
        exit()
    qtrim_min = x.quantile(trim)
    qtrim_mid = x.quantile(0.5)
    qtrim_max = x.quantile(1-trim)
    if trim < 0.5:
        y[x < qtrim_min] = qtrim_min
        y[x > qtrim_max] = qtrim_max
    else:
        y[x != None] = qtrim_mid
    return y


def stand(z, trim_num):
    x = win(z, trim_num)
    x_mean = np.nanmean(x)
    x_std = np.nanstd(x)
    y = (x - x_mean) / x_std
    return y


def std_winsor(z):
    tmp_z = z.copy()
    tmp_z.dropna()
    z_std = np.std(tmp_z)
    min_std = -3 * z_std
    max_std = 3 * z_std
    z[z<min_std] = min_std
    z[z>max_std] = max_std
    z[z==None] = min_std
    return z


def save_obj(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_obj(file):
    with open(file, 'rb') as f:
        obj = pkl.load(f)
    return obj

def dict_prod(source, target):
    assert source.keys() == target.keys(), print(set(source.keys()).difference(set(target.keys())))
    return sum(source.get(key, 0) * target.get(key, 0) for key in source.keys() | target.keys())


def select_suspend_stock(all_stock, hs300_stock):
    all_stock = set(all_stock)
    hs300_stock = set(hs300_stock)
    suspend_stock = hs300_stock.difference(all_stock)
    return suspend_stock


def explore_data():
    raw_factor = pd.read_csv(raw_factor_path, sep=',')
    hs300_idx_df = pd.read_csv(hs300_wgt_path, sep=',')
    hs300_trade_datas = pd.read_csv(hs300_trade_data_path, sep='\t')
    hs300_all_datas = pd.read_csv(hs300_all_data_path, sep='\t')
    date_list = raw_factor['Date'].drop_duplicates().to_list()
    for date in date_list:
        hs300_idx_code = hs300_idx_df[hs300_idx_df.Date == date]['sec_code']
        hs300_trade_date = hs300_trade_datas[hs300_trade_datas.trade_date == date]['ts_code']
        hs300_all_data = hs300_all_datas[hs300_all_datas.Date == date]['Codes']
        print(len(hs300_idx_code))
        print(len(hs300_trade_date))
        print(len(hs300_all_data))
        print(len(set(hs300_all_data.to_list()) & set(hs300_trade_date.to_list())))
        print(len(set(hs300_idx_code.to_list()) & set(hs300_trade_date.to_list())))
        exit()


def get_hs300_name2code(hs300_wgt_data):
    file_name = 'hs300_name2code.pkl'
    hs300_name2code_file_path = os.path.join(tmp_path, file_name)
    if os.path.exists(hs300_name2code_file_path):
        with open(hs300_name2code_file_path, 'rb') as f:
            name2code = pkl.load(f)
    else:
        name2code = {}
        all_code = hs300_wgt_data['ts_code'].drop_duplicates().to_list()
        for code in all_code:
            stock_name = hs300_wgt_data[hs300_wgt_data.ts_code == code]['stock_name']
            stock_name = stock_name.iloc[0]
            name2code[stock_name] = code
        with open(hs300_name2code_file_path, 'wb') as f:
            pkl.dump(name2code, f)
    return name2code



if __name__ == '__main__':
    prepare_hs300_wgt_df()
