import pandas as pd
from tqdm import tqdm
import pickle as pkl
import os

hs300_trading_monthly_path = r'./data/HS300tradingmonthly.txt'
raw_factor_path = r'./data/raw_factor_test.csv'
hs300_trade_data_path = r'./data/399300.SZ_tradingdata_part1.csv'
hs300_wgt_path = r'./data/HS300_idx_wt.csv'
hs300_idx_codes_path = r'./data/399300.SZ_weight.txt'
hs300_all_data_path = r'./data/HS300alldata_vol2.txt'
tmp_path = r'./data/tmp'

industry_upper_bound = 0.01
industry_lower_bound = 0.01

factor_upper_bound = 0.35
factor_lower_bound = 0.00

stock_upper_bound = 0.01
stock_lower_bound = 0.01

constr_factor = ['Size_Factor', 'IdioVolatility_Factor', 'RSI_Factor', 'Quality_Factor', 'Value_Factor']


def create_stock_dict(stock_codes):
    stock_codes.sort()
    stock2id = {}
    with open('./data/hs300_stock2id.txt', 'w') as f:
        for idx, code in enumerate(stock_codes):
            stock2id[code] = idx
            f.write('\t'.join([code, str(idx)]) + '\n')
    id2stock = dict([(val, key) for key, val in stock2id.items()])
    with open('./data/hs300_stock2id.pkl','wb') as f1, open('./data/hs300_id2stock.pkl','wb') as f2:
        pkl.dump(stock2id, f1)
        pkl.dump(id2stock, f2)


def dict_prod(source, target):
    assert source.keys() == target.keys(), print(set(source.keys()).difference(set(target.keys())))
    return sum(source.get(key, 0) * target.get(key, 0) for key in source.keys() | target.keys())


def prepare_data( ):
    split_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg']
    trade_data_df = pd.read_csv(hs300_trade_data_path, sep='\t')
    wgt_data_df = pd.read_csv(hs300_wgt_path, sep=',')
    trade_data_df = trade_data_df[split_columns]
    all_data_df = wgt_data_df.merge(trade_data_df, how='left',
                                    left_on=['sec_code', 'Date'], right_on=['ts_code', 'trade_date'])
    all_data_df.drop(['ts_code', 'trade_date'], axis=1)
    all_data_df.to_csv('tmp.csv', sep=',')
    print(all_data_df.head)


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


def get_hs300_datelist(hs300_wgt_data):
    return hs300_wgt_data['trade_date'].drop_duplicates().to_list()


def prepare_hs300_wgt_df():
    destination = tmp_path
    if not os.path.exists(destination):
        os.makedirs(destination)
    hs300_wgt_data = pd.read_csv(hs300_wgt_path, sep=',')
    rename_dict = {
        'sec_code': "ts_code",
        'sec_name': 'stock_name',
        'Date': 'trade_date'}
    hs300_wgt_data = hs300_wgt_data.drop(columns='Adjust date')
    hs300_wgt_data = hs300_wgt_data.rename(columns=rename_dict)
    name2code = get_hs300_name2code(hs300_wgt_data)
    code2name = dict([(val, key) for key, val in name2code.items()])
    date_list = get_hs300_datelist(hs300_wgt_data)
    date_list.sort()
    code_set = set([])
    hs300_col_name = hs300_wgt_data.columns.to_list()
    nan_data = pd.Series(index=hs300_col_name)
    hs300_wgt_df = pd.DataFrame(columns=hs300_col_name)
    for date in date_list:
        day_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        day_code_list = day_wgt_data['ts_code'].to_list()
        day_code_list = set(day_code_list)
        code_set.update(set(day_code_list))
        day_diff_code_list = code_set.difference(day_code_list)
        # print('day {} add '.format(date), len(day_diff_code_list))
        hs300_wgt_df = hs300_wgt_df.append(day_wgt_data, ignore_index=True)
        if day_diff_code_list:
            nan_col_code = list(day_diff_code_list)
            nan_col_name = [code2name[i] for i in day_diff_code_list]
            nan_dict = {
                'ts_code': nan_col_code,
                'stock_name': nan_col_name,
                'weight': 0,
                'trade_date': date
            }
            nan_df = pd.DataFrame(nan_dict)
            hs300_wgt_df = hs300_wgt_df.append(nan_df, ignore_index=True)
            # print('add {} {} on the date {}'.format(code, code2name[code], date))
    hs300_wgt_df.to_csv(os.path.join(destination, 'hs300_wgt_df.csv'), sep=',')
    return hs300_wgt_df


if __name__ == '__main__':
    prepare_hs300_wgt_df()
