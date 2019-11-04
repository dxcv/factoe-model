import pandas as pd
import numpy as np
import copy
from time import time
from tqdm import tqdm
import pickle as pkl
import os

tmp_path = r'./data/tmp'
rename_dict = {
    'Codes': "ts_code",
    'GicsCodes': "gics_code",
    'Gics': 'industry_name',
    'Date': 'trade_date',
    'sec_code': "ts_code",
    'f_ann_date': "trade_date",
    'sec_name': 'stock_name'}


def norm_df(hs300_wgt_data):
    hs300_wgt_data = hs300_wgt_data.rename(columns=rename_dict)
    hs300_wgt_data = hs300_wgt_data.replace({'ts_code': '000022.SZ'}, '001872.SZ')
    return hs300_wgt_data


def read_csv(sr, sep=','):
    df = pd.read_csv(sr, sep=sep)
    df = norm_df(df)
    return df


def save_obj(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_obj(file):
    with open(file,'rb') as f:
        obj = pkl.load(f)
    return obj


def get_gics_code2gics_info():
    file_name = r'./data/tmp/code2gics_info.pkl'
    if os.path.exists(file_name):
        code2gics_info = load_obj(file_name)
        return code2gics_info
    hs300_industry_df = prepare_hs300_industry_data_df()
    code = hs300_industry_df['ts_code'].to_list()
    industry_name = hs300_industry_df['industry_name'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], (i[1], i[2])) for i in zip(code, industry_name, gics_code)]
    tuple_set = set(tuple_list)
    code2gics_info = dict(list(tuple_set))
    save_obj(file_name, code2gics_info)
    return code2gics_info


def get_code2gics():
    file_name = r'./data/tmp/code2gics.pkl'
    if os.path.exists(file_name):
        code2gics_info = load_obj(file_name)
        return code2gics_info
    hs300_industry_df = prepare_hs300_industry_data_df()
    code = hs300_industry_df['ts_code'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], i[1]) for i in zip(code, gics_code)]
    tuple_set = set(tuple_list)
    code2gics = dict(list(tuple_set))
    save_obj(file_name, code2gics)
    return code2gics


def get_gics2name():
    file_name = r'./data/tmp/gics2name.pkl'
    if os.path.exists(file_name):
        gics2name = load_obj(file_name)
        return gics2name
    hs300_industry_df = prepare_hs300_industry_data_df()
    industry_name = hs300_industry_df['industry_name'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], i[1]) for i in zip(gics_code, industry_name)]
    tuple_set = set(tuple_list)
    gics2name = dict(list(tuple_set))
    save_obj(file_name, gics2name)
    return gics2name


def get_name2code(wgt_data):
    file_name = 'hs300_name2code.pkl'
    name2code_file_path = os.path.join(tmp_path, file_name)
    if os.path.exists(name2code_file_path):
        name2code = load_obj(name2code_file_path)
        return name2code
    name2code = {}
    all_code = wgt_data['ts_code'].drop_duplicates().to_list()
    for code in all_code:
        stock_name = wgt_data[wgt_data.ts_code == code]['stock_name']
        stock_name = stock_name.iloc[0]
        name2code[stock_name] = code
    save_obj(name2code_file_path, name2code)
    return name2code


def prepare_hs300_industry_data_df():
    hs300_industry_ts_path = r'./data/HS300_industry_ts.txt'
    hs300_industry_path = r'./data/HS300_industry.txt'
    file_name = './data/tmp/hs300_industry.csv'
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    hs300_industry_ts = pd.read_csv(hs300_industry_ts_path, encoding='gbk')
    hs300_industry = pd.read_table(hs300_industry_path, encoding='gbk', sep=',')
    for i in range(hs300_industry_ts.shape[0]):
        if np.isnan(hs300_industry_ts.loc[i, 'GicsCodes']):
            hs300_industry_ts.loc[i, 'Gics'] = list(
                hs300_industry.loc[hs300_industry.Codes == hs300_industry_ts.loc[i, 'Codes'], 'industry_gics'])
            hs300_industry_ts.loc[i, 'GicsCodes'] = list(
                hs300_industry.loc[hs300_industry.Codes == hs300_industry_ts.loc[i, 'Codes'], 'industry_gicscode'])
    hs300industry_ts = hs300_industry_ts.rename(columns=rename_dict)
    hs300industry_ts.to_csv(file_name, index=False)
    return hs300industry_ts


def prepare_hs300_wgt_data_df(hs300_wgt_data):
    hs300_wgt_df_path = r'./data/tmp/hs300_wgt_df.csv'
    if os.path.exists(hs300_wgt_df_path):
        return pd.read_csv(hs300_wgt_df_path)
    hs300_industry_df = prepare_hs300_industry_data_df()
    code2gics_info = get_gics_code2gics_info()
    hs300_wgt_data = hs300_wgt_data.merge(hs300_industry_df, how='left', on=['ts_code', 'trade_date'])
    name2code = get_name2code(hs300_wgt_data)
    code2name = dict([(val, key) for key, val in name2code.items()])
    code2gics_info['001872.SZ'] = copy.deepcopy(code2gics_info['000022.SZ'])
    date_list = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
    date_list.sort()
    code_set = set([])
    hs300_col_name = hs300_wgt_data.columns.to_list()
    hs300_wgt_df = pd.DataFrame(columns=hs300_col_name)
    for date in tqdm(date_list):
        day_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        day_code_list = day_wgt_data['ts_code'].to_list()
        day_gics_list = [code2gics_info[i][1] for i in day_code_list]
        day_wgt_data['gics_code'] = day_gics_list
        assert len(day_code_list) == len(set(day_code_list))
        code_set.update(set(day_code_list))
        day_diff_code_list = code_set.difference(day_code_list)
        hs300_wgt_df = hs300_wgt_df.append(day_wgt_data, ignore_index=True)
        if day_diff_code_list:
            nan_col_code = list(day_diff_code_list)
            nan_col_name = [code2name[i] for i in day_diff_code_list]
            nan_col_industry_name = [code2gics_info[i][0] for i in day_diff_code_list]
            nan_col_gics_code = [code2gics_info[i][1] for i in day_diff_code_list]
            nan_dict = {
                'ts_code': nan_col_code,
                'stock_name': nan_col_name,
                'gics_code': nan_col_gics_code,
                'industry_name': nan_col_industry_name,
                'weight': 0,
                'trade_date': date
            }
            nan_df = pd.DataFrame(nan_dict)
            hs300_wgt_df = hs300_wgt_df.append(nan_df, ignore_index=True)
    hs300_wgt_df.to_csv(hs300_wgt_df_path, index=False)
    return hs300_wgt_df


def prepare_all_date_trading_data_df(hs300_trading_data):
    file_name = './data/tmp/all_trading_data.csv'
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    ts_code = hs300_trading_data['ts_code'].drop_duplicates().to_list()
    trade_date = hs300_trading_data['trade_date'].drop_duplicates().to_list()
    trade_date.sort()
    df = pd.DataFrame()
    for code in tqdm(ts_code):
        col_dict = {
            'ts_code': code,
            'trade_date': trade_date
        }
        tmp_df = pd.DataFrame(col_dict)
        df = df.append(tmp_df, ignore_index=True)
    df = df.merge(hs300_trading_data, how='left', on=['ts_code', 'trade_date'])
    df = df.fillna(method='ffill')
    df.to_csv(file_name, index=False)
    return df


def prepare_all_date_monthly_data_df():
    hs300_all_monthly_date_df_path = r'./data/tmp/hs300_all_monthly_date_df.csv'
    if os.path.exists(hs300_all_monthly_date_df_path):
        return pd.read_csv(hs300_all_monthly_date_df_path)
    hs300_industry_data = prepare_hs300_industry_data_df()
    date_list = hs300_industry_data['trade_date'].drop_duplicates().to_list()
    date_list.sort()
    code_list = hs300_industry_data['ts_code'].drop_duplicates().to_list()
    code2gics_info = get_gics_code2gics_info()
    df = pd.DataFrame()
    for code in tqdm(code_list):
        industry_name = code2gics_info[code][0]
        gics = code2gics_info[code][1]
        tmp_df = pd.DataFrame(
            {
                'ts_code': code,
                'trade_date': date_list,
                'gics_code': gics,
                'industry_name': industry_name
            }
        )
        df = df.append(tmp_df)
    df.to_csv(hs300_all_monthly_date_df_path, index=False)
    return df


def prepare_monthly_price_change_data(hs300_trading_monthly, month_const, lag_num):
    monthly_price_change_data_path = r'./data/tmp/monthly_price_change_data.csv'
    if os.path.exists(monthly_price_change_data_path):
        return pd.read_csv(monthly_price_change_data_path)
    hs300_trading_monthly = hs300_trading_monthly[['ts_code', 'trade_date', 'close_hfq', 'pre_close_hfq']]
    hs300_trading_monthly = hs300_trading_monthly.sort_values(['ts_code', 'trade_date'])
    code_list = hs300_trading_monthly['ts_code'].drop_duplicates().to_list()
    df = pd.DataFrame()
    for code in tqdm(code_list):
        code_hs300_trading_monthly = hs300_trading_monthly[hs300_trading_monthly.ts_code == code]
        date_list = code_hs300_trading_monthly['trade_date'].to_list()
        close_hfq = code_hs300_trading_monthly['close_hfq'].to_numpy()
        pre_close_hfq = code_hs300_trading_monthly['pre_close_hfq'].to_numpy()
        last_12mon_chg = []
        last_1mon_chg = []
        last_1mon_return = []
        for i in range(1, len(close_hfq)):
            pre_1mon = pre_close_hfq[i - 1]
            tmp_1mon_chg = pre_close_hfq[i] / pre_1mon - 1 if pre_1mon else 0
            last_1mon_chg.append(tmp_1mon_chg)
            tmp_last_1mon_return = close_hfq[i] / close_hfq[i-1] - 1 if close_hfq[i-1] else 0
            last_1mon_return.append(tmp_last_1mon_return)
            if month_const <= i:
                pre_12mon = close_hfq[i-month_const]
                tmp_12mon_chg = close_hfq[i] / pre_12mon - 1 if pre_12mon else 0
                last_12mon_chg.append(tmp_12mon_chg)
        last_12mon_chg_lag = copy.deepcopy(last_12mon_chg[:-lag_num])
        last_12mon_chg_lag = [np.nan] * (month_const + lag_num) + last_12mon_chg_lag
        last_12mon_chg = [np.nan] * month_const + last_12mon_chg
        last_1mon_chg = [np.nan] + last_1mon_chg
        last_1mon_return = [np.nan] + last_1mon_return
        assert len(last_12mon_chg_lag) == len(last_12mon_chg) == len(close_hfq) == \
               len(last_1mon_chg) == len(last_1mon_return)
        tmp_df = pd.DataFrame(
            {
                'ts_code':code,
                'trade_date':date_list,
                'last_12mon_pricechange':last_12mon_chg,
                'last_12mon_pricechange_lag': last_12mon_chg_lag,
                '1mon_tradingreturn':last_1mon_return,
                'last_1mon_pricechange':last_1mon_chg
            }
        )
        df = df.append(tmp_df)
    df.to_csv(monthly_price_change_data_path,index=False)
    return df


def prepare_full_final_sheet(hs300_balance_sheet, hs300_income_sheet,
                       hs300_cashflow_sheet, hs300_trading_data):
    blank_df_path = r'./data/tmp/blank_df.csv'
    hs300_balance_sheet = hs300_balance_sheet.sort_values(['ts_code', 'ann_date', 'trade_date', 'end_date'])
    hs300_income_sheet = hs300_income_sheet.sort_values(['ts_code', 'ann_date', 'trade_date', 'end_date'])
    hs300_cashflow_sheet = hs300_cashflow_sheet.sort_values(['ts_code', 'ann_date', 'trade_date', 'end_date'])
    hs300_balance_sheet = hs300_balance_sheet.drop(['end_date', 'ann_date'], axis=1)
    hs300_income_sheet = hs300_income_sheet.drop(['end_date', 'ann_date'], axis=1)
    hs300_cashflow_sheet = hs300_cashflow_sheet.drop(['end_date', 'ann_date'], axis=1)
    hs300_balance_sheet = hs300_balance_sheet.drop_duplicates(subset='trade_date', keep="last")
    hs300_income_sheet = hs300_income_sheet.drop_duplicates(subset='trade_date', keep="last")
    hs300_cashflow_sheet = hs300_cashflow_sheet.drop_duplicates(subset='trade_date', keep="last")
    if os.path.exists(blank_df_path):
        df = pd.read_csv(blank_df_path)
    else:
        ts_code = hs300_trading_data['ts_code'].drop_duplicates().to_list()
        trade_date = hs300_trading_data['trade_date'].to_list()
        tmp_date = hs300_trading_data['trade_date'].to_list()
        tmp_date += hs300_income_sheet['trade_date'].to_list()
        tmp_date += hs300_cashflow_sheet['trade_date'].to_list()
        trade_date += tmp_date
        trade_date = list(set(trade_date))
        trade_date.sort()
        df = pd.DataFrame()
        for code in tqdm(ts_code):
            col_dict = {
                'ts_code': code,
                'trade_date': trade_date
            }
            tmp_df = pd.DataFrame(col_dict)
            df = df.append(tmp_df, ignore_index=True)
        df.to_csv(blank_df_path, index=False)
    df = df.merge(hs300_balance_sheet, how='left', on=['ts_code', 'trade_date'])
    df = df.merge(hs300_income_sheet, how='left', on=['ts_code', 'trade_date'])
    df = df.merge(hs300_cashflow_sheet, how='left', on=['ts_code', 'trade_date'])
    df = df.fillna(method='ffill')
    df.to_csv('all_data_sheet.csv', index=False)
    return df


def prepare_indicator_sheet(hs300_income_sheet, hs300_trading_data, name='indicator'):
    blank_df_path = r'./data/tmp/blank_df_{}.csv'.format(name)
    hs300_income_sheet = hs300_income_sheet.sort_values(['ts_code', 'ann_date', 'end_date'])
    hs300_income_sheet = hs300_income_sheet.rename(columns={'ann_date': 'trade_date'})
    hs300_income_sheet = hs300_income_sheet.drop(['end_date'], axis=1)
    hs300_income_sheet = hs300_income_sheet.drop_duplicates(subset='trade_date', keep="last")
    if os.path.exists(blank_df_path):
        df = pd.read_csv(blank_df_path)
    else:
        ts_code = hs300_trading_data['ts_code'].drop_duplicates().to_list()
        trade_date = hs300_trading_data['trade_date'].to_list()
        tmp_date = hs300_income_sheet['trade_date'].drop_duplicates().to_list()
        tmp_date = [i for i in tmp_date if not np.isnan(i)]
        trade_date += tmp_date
        trade_date = list(set(trade_date))
        trade_date.sort()
        df = pd.DataFrame()
        for code in tqdm(ts_code):
            col_dict = {
                'ts_code': code,
                'trade_date': trade_date
            }
            tmp_df = pd.DataFrame(col_dict)
            df = df.append(tmp_df, ignore_index=True)
        df.to_csv(blank_df_path, index=False)
    df = df.merge(hs300_income_sheet, how='left', on=['ts_code', 'trade_date'])
    df = df.fillna(method='ffill')
    return df


def prepare_vol_data_df(hs300_trading_data, vol_lag):
    vol_data_path = r'./data/tmp/vol_data.csv'
    if os.path.exists(vol_data_path):
        return pd.read_csv(vol_data_path)
    hs300_trading_data = hs300_trading_data[['trade_date', 'ts_code', 'pct_chg_hfq']]
    code_list = hs300_trading_data['ts_code'].drop_duplicates().to_list()
    vol_data = pd.DataFrame()
    for code in tqdm(code_list):
        code_trading_data = hs300_trading_data[hs300_trading_data.ts_code == code]
        code_trading_data = code_trading_data.sort_values(['trade_date'])
        pct_chg_hfq = code_trading_data['pct_chg_hfq'].to_list()
        trade_date = code_trading_data['trade_date'].to_list()
        vol = []
        for i in range(vol_lag, len(pct_chg_hfq)):
            tmp = pct_chg_hfq[i-vol_lag:i]
            tmp = np.nanstd(tmp)
            vol.append(tmp)
        vol = [np.nan] * vol_lag + vol
        assert len(vol) == len(pct_chg_hfq)
        tmp_vol_data = pd.DataFrame({
            'ts_code': code,
            'trade_date': trade_date,
            'Volatility': vol
        })
        vol_data = vol_data.append(tmp_vol_data)
    vol_data.to_csv(vol_data_path, index=False)
    return vol_data


def prepare_rsi_data_df(hs300_trading_data, rsi_lag):
    rsi_data_path = r'./data/tmp/rsi_data.csv'
    if os.path.exists(rsi_data_path):
        return pd.read_csv(rsi_data_path)
    hs300_trading_data = hs300_trading_data[['trade_date', 'ts_code', 'pct_chg_hfq']]
    code_list = hs300_trading_data['ts_code'].drop_duplicates().to_list()
    rsi_data = pd.DataFrame()
    for code in tqdm(code_list):
        code_trading_data = hs300_trading_data[hs300_trading_data.ts_code == code]
        code_trading_data = code_trading_data.sort_values(['trade_date'])
        pct_chg_hfq = code_trading_data['pct_chg_hfq'].to_list()
        trade_date = code_trading_data['trade_date'].to_list()
        rsi = []
        for i in range(rsi_lag, len(pct_chg_hfq)):
            tmp = np.array(pct_chg_hfq[i - rsi_lag:i])
            m_gain = sum(tmp[tmp>0])
            m_loss = np.abs(sum(tmp[tmp<0]))
            tmp = (m_gain/(m_gain + m_loss)) * 100
            rsi.append(tmp)
        rsi = [np.nan] * rsi_lag + rsi
        assert len(rsi) == len(pct_chg_hfq)
        tmp_rsi_data = pd.DataFrame({
            'ts_code': code,
            'trade_date': trade_date,
            'RSI': rsi
        })
        rsi_data = rsi_data.append(tmp_rsi_data)
    rsi_data.to_csv(rsi_data_path, index=False)
    return rsi_data


def main():
    start = time()
    month_const = 12
    lag_num = 2
    vol_lag = 60
    rsi_lag = 10
    hs300_trade_data_path = r'./data/399300.SZ_tradingdata.txt'
    hs300_wgt_path = r'./data/HS300_idx_wt.csv'
    balance_data_path = r'./data/399300.SZ_balancesheetdata.txt'
    income_data_path = r'./data/399300.SZ_incomedata.txt'
    cashflow_data_path = r'./data/399300.SZ_cashflowdata.txt'
    fina_indicator_data_path = r'./data/399300.SZ_fina_indicatordata.txt'

    hs300_trading_data = read_csv(hs300_trade_data_path, '\t')
    hs300_trading_data = hs300_trading_data.sort_values(['ts_code', 'trade_date'])
    hs300_wgt_data = read_csv(hs300_wgt_path)
    hs300_trading_data = prepare_all_date_trading_data_df(hs300_trading_data)
    hs300_all_monthly_data = prepare_all_date_monthly_data_df()
    hs300_all_monthly_data = norm_df(hs300_all_monthly_data)
    hs300_all_monthly_trading_date = hs300_all_monthly_data.merge(hs300_trading_data,
                                                                  how='left', on=['ts_code', 'trade_date'])
    hs300_monthly_chg_data = prepare_monthly_price_change_data(hs300_all_monthly_trading_date, month_const, lag_num)
    hs300_all_monthly_trading_date = hs300_all_monthly_trading_date.merge(hs300_monthly_chg_data,
                                                                          how='left', on=['ts_code', 'trade_date'])
    hs300_all_monthly_trading_date["12monPC_1monPC"] = hs300_all_monthly_trading_date["last_12mon_pricechange_lag"] - \
                                                   hs300_all_monthly_trading_date["last_1mon_pricechange"]
    hs300_all_monthly_trading_date.to_csv('./data/tmp/monthly_data.csv', index=False)
    hs300_balancesheetdata = read_csv(balance_data_path, sep='\t')
    hs300_incomedata = read_csv(income_data_path, sep='\t')
    hs300_cashflowdata = read_csv(cashflow_data_path, sep='\t')
    hs300_fina_indicatordata = read_csv(fina_indicator_data_path, sep='\t')
    vol_data = prepare_vol_data_df(hs300_trading_data, vol_lag)
    rsi_data = prepare_rsi_data_df(hs300_trading_data, rsi_lag)
    all_sheet = prepare_full_final_sheet(hs300_balancesheetdata, hs300_incomedata,
                                         hs300_cashflowdata, hs300_trading_data)
    hs300_fina_indicatordata_ = prepare_indicator_sheet(hs300_fina_indicatordata, hs300_trading_data, 'indicator')
    liab_and_mv = hs300_trading_data[['ts_code', 'trade_date', 'total_mv', 'pb', 'close']]
    hs300_wgt_data = prepare_hs300_wgt_data_df(hs300_wgt_data)
    hs300_wgt_data = hs300_wgt_data.merge(vol_data, how='left', on=['ts_code', 'trade_date'])
    hs300_wgt_data = hs300_wgt_data.merge(rsi_data, how='left', on=['ts_code', 'trade_date'])
    hs300_wgt_data = hs300_wgt_data.merge(all_sheet, how='left', on=['ts_code', 'trade_date'])
    hs300_wgt_data = hs300_wgt_data.merge(hs300_fina_indicatordata_, how='left', on=['ts_code', 'trade_date'])
    hs300_wgt_data = hs300_wgt_data.merge(liab_and_mv, how='left', on=['ts_code', 'trade_date'])
    hs300_wgt_data["Log_mkt_Cap"] = np.log(list(hs300_wgt_data["total_mv"]))
    hs300_wgt_data["Cash_Over_MktCap"] = hs300_wgt_data.fcff.div(
        hs300_wgt_data.total_mv, axis=0)
    hs300_wgt_data["Rev_Over_mktCap"] = hs300_wgt_data.revenue.div(
        hs300_wgt_data.total_mv, axis=0)
    hs300_wgt_data["NOCF_Over_Debt"] = hs300_wgt_data.n_cashflow_act.div(
        hs300_wgt_data.total_liab, axis=0)
    hs300_wgt_data.to_csv('./data/tmp/hs300_all_trading_data_monthly.csv', index=False)
    print('cost time:', time()-start)


if __name__ == '__main__':
    main()
