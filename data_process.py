import os
import copy
import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from tqdm import tqdm
import pickle as pkl
import bisect
import math
from pprint import pprint
from configuration import Config

config_data_path = r'./config/config.json'
config = Config(config_data_path)
tmp_path = config.tmp_path
rename_dict = config.rename_dict


def norm_df(hs300_wgt_data):
    hs300_wgt_data = hs300_wgt_data.rename(columns=rename_dict)
    hs300_wgt_data = hs300_wgt_data.replace({'ts_code': '000022.SZ'}, '001872.SZ')
    return hs300_wgt_data


def read_csv(sr, sep=',', encoding='utf8'):
    df = pd.read_csv(sr, sep=sep, encoding=encoding)
    df = norm_df(df)
    return df


def save_obj(file, obj):
    with open(file, 'wb') as f:
        pkl.dump(obj, f)


def load_obj(file):
    with open(file, 'rb') as f:
        obj = pkl.load(f)
    return obj


def merge_data(df1, df2):
    df1 = df1.merge(df2, how='left', on=['ts_code', 'trade_data'])
    return df1


def get_gics_code2gics_info():
    file_path =config.code2gics_info
    if os.path.exists(file_path):
        code2gics_info = load_obj(file_path)
        return code2gics_info
    hs300_industry_df = prepare_hs300_industry_data_df()
    code = hs300_industry_df['ts_code'].to_list()
    industry_name = hs300_industry_df['industry_name'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], (i[1], i[2])) for i in zip(code, industry_name, gics_code)]
    tuple_set = set(tuple_list)
    code2gics_info = dict(list(tuple_set))
    save_obj(file_path, code2gics_info)
    return code2gics_info


def get_code2gics():
    file_path = config.code2gics
    if os.path.exists(file_path):
        code2gics_info = load_obj(file_path)
        return code2gics_info
    hs300_industry_df = prepare_hs300_industry_data_df()
    code = hs300_industry_df['ts_code'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], i[1]) for i in zip(code, gics_code)]
    tuple_set = set(tuple_list)
    code2gics = dict(list(tuple_set))
    save_obj(file_path, code2gics)
    return code2gics


def get_gics2name():
    file_path = config.gics2name
    if os.path.exists(file_path):
        gics2name = load_obj(file_path)
        return gics2name
    hs300_industry_df = prepare_hs300_industry_data_df()
    industry_name = hs300_industry_df['industry_name'].to_list()
    gics_code = hs300_industry_df['gics_code'].to_list()
    tuple_list = [(i[0], i[1]) for i in zip(gics_code, industry_name)]
    tuple_set = set(tuple_list)
    gics2name = dict(list(tuple_set))
    save_obj(file_path, gics2name)
    return gics2name


def get_name2code(wgt_data):
    file_path = config.name2code
    if os.path.exists(file_path):
        name2code = load_obj(file_path)
        return name2code
    name2code = {}
    all_code = wgt_data['ts_code'].drop_duplicates().to_list()
    for code in all_code:
        stock_name = wgt_data[wgt_data.ts_code == code]['stock_name']
        stock_name = stock_name.iloc[0]
        name2code[stock_name] = code
    save_obj(file_path, name2code)
    return name2code


def prepare_hs300_industry_data_df():
    hs300_industry_ts_path = config.industry_ts_path
    hs300_industry_path = config.industry_path
    file_name = config.fixed_industry_data_path
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    industry_ts = read_csv(hs300_industry_ts_path, encoding='gbk')
    industry = read_csv(hs300_industry_path, encoding='gbk')
    tmp_code_list = industry['ts_code']
    tmp_gics_list = industry['gics_code']
    tmp_industry_name_list = industry['industry_name']
    code2gics = dict(zip(tmp_code_list, tmp_gics_list))
    code2industry_name = dict(zip(tmp_code_list, tmp_industry_name_list))
    num = industry_ts.shape[0]
    for i in tqdm(range(num)):
        if np.isnan(industry_ts.loc[i, 'gics_code']):
            tmp = industry_ts.loc[i]
            code = tmp['ts_code']
            industry_ts.loc[i, ['industry_name', 'gics_code']] = [code2industry_name[code], code2gics[code]]
            # industry_ts.loc[i, 'gics_code'] = code2industry_name[code]
    industry_ts.to_csv(file_name, index=False)
    return industry_ts


def prepare_fix_wgt_data_df(hs300_wgt_data):
    hs300_wgt_df_path = config.fixed_weight_data_path
    if os.path.exists(hs300_wgt_df_path):
        return pd.read_csv(hs300_wgt_df_path)
    hs300_industry_df = prepare_hs300_industry_data_df()
    code2gics_info = get_gics_code2gics_info()
    hs300_wgt_data = hs300_wgt_data.merge(hs300_industry_df, how='left', on=['ts_code', 'trade_date'])
    name2code = get_name2code(hs300_wgt_data)
    code2name = dict([(val, key) for key, val in name2code.items()])
    # code2gics_info['001872.SZ'] = copy.deepcopy(code2gics_info['000022.SZ'])
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


def prepare_fixed_daily_data(daily_data):
    file_path = config.fixed_daily_data_path
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    ts_code = daily_data['ts_code'].drop_duplicates().to_list()
    trade_date = daily_data['trade_date'].drop_duplicates().to_list()
    trade_date.sort()
    df = pd.DataFrame()
    for code in tqdm(ts_code):
        col_dict = {'ts_code': code, 'trade_date': trade_date}
        tmp_df = pd.DataFrame(col_dict)
        tmp_df = tmp_df.merge(daily_data, how='left', on=['ts_code', 'trade_date'])
        tmp_df = tmp_df.fillna(method='ffill')
        df = df.append(tmp_df, ignore_index=True)
    df.to_csv(file_path, index=False)
    return df


def prepare_fixed_monthly_data_df(opt_date_list, trade_date_list, code_list):
    fixed_opt_monthly_data_path = config.fixed_opt_monthly_data_path
    fixed_trade_monthly_data_path = config.fixed_trade_monthly_data_path
    if os.path.exists(fixed_opt_monthly_data_path) and os.path.exists(fixed_trade_monthly_data_path):
        return pd.read_csv(fixed_opt_monthly_data_path), pd.read_csv(fixed_trade_monthly_data_path)
    opt_date_list = sorted(opt_date_list)
    trade_date_list = sorted(trade_date_list)
    code2gics = get_code2gics()
    gics2name = get_gics2name()
    opt_df = pd.DataFrame()
    trade_df = pd.DataFrame()
    for code in tqdm(code_list):
        gics = code2gics[code]
        industry_name = gics2name[gics]
        tmp_opt_df = pd.DataFrame(
            {'ts_code': code, 'trade_date': opt_date_list,
             'gics_code': gics, 'industry_name': industry_name
            }
        )
        tmp_trade_df = pd.DataFrame(
            {'ts_code': code, 'trade_date': trade_date_list,
             'gics_code': gics, 'industry_name': industry_name
            }
        )
        opt_df = opt_df.append(tmp_opt_df)
        trade_df = trade_df.append(tmp_trade_df)
    opt_df.to_csv(fixed_opt_monthly_data_path, index=False)
    trade_df.to_csv(fixed_trade_monthly_data_path, index=False)
    return opt_df, trade_df


def prepare_monthly_price_change_data(monthly_data, month_const, lag_num, sava_path):
    monthly_price_change_data_path = sava_path
    if os.path.exists(monthly_price_change_data_path):
        return pd.read_csv(monthly_price_change_data_path)
    monthly_data = monthly_data[['ts_code', 'trade_date', 'close_hfq', 'pre_close_hfq']]
    monthly_data = monthly_data.sort_values(['ts_code', 'trade_date'])
    code_list = monthly_data['ts_code'].drop_duplicates().to_list()
    df = pd.DataFrame()
    for code in tqdm(code_list):
        code_hs300_trading_monthly = monthly_data[monthly_data.ts_code == code]
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
                '1mon_tradingreturn': last_1mon_return,
                'last_1mon_pricechange': last_1mon_chg
            }
        )
        df = df.append(tmp_df)
    df.to_csv(monthly_price_change_data_path, index=False)
    return df


def prepare_full_financial_sheet(hs300_balance_sheet, hs300_income_sheet,
                                 hs300_cashflow_sheet, hs300_trading_data, save_path):
    blank_df_path = save_path
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
    # df.to_csv('all_data_sheet.csv', index=False)
    return df


def prepare_indicator_sheet(hs300_income_sheet, hs300_trading_data, blank_df_path):
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
    vol_data_path = config.vol_data
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
    rsi_data_path = config.rsi_data
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


def prepare_one_day_late_trade_date_list(daily_date, opt_date_list):
    all_trade_date_list = daily_date['trade_date'].drop_duplicates().to_list()
    all_trade_date_list.sort()
    trade_date_list = []
    opt2trade = {}
    for opt_date in opt_date_list:
        idx = all_trade_date_list.index(opt_date)
        trade_date = all_trade_date_list[idx+1]
        trade_date_list.append(trade_date)
        opt2trade[opt_date] = trade_date
    save_obj(config.opt2trade, opt2trade)
    return trade_date_list, opt2trade


def prepare_week_trade_date_list(daily_data, week=2):
    date_list = daily_data[['trade_date']].drop_duplicates()
    date_list = date_list.astype({'trade_date': str})
    week_date = date_list[date_list.apply(
        lambda x: datetime.strptime(x['trade_date'], "%Y%m%d").weekday() == week, axis=1)]
    week_date = week_date.astype({'trade_date': int})
    week_date = week_date['trade_date'].to_list()
    week_date = [i for i in week_date if i >= 20050429]
    return week_date


def prepare_week_fix_wgt_data_df(hs300_wgt_data, opt_date_list, daily_data):
    fixed_opt_weight_data_path = config.fixed_opt_weight_data_path
    if os.path.exists(fixed_opt_weight_data_path):
        return pd.read_csv(fixed_opt_weight_data_path)
    date2codelist = {}
    date_list = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
    date_list.sort()
    for date in date_list:
        tmp = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        tmp = tmp['ts_code'].drop_duplicates().to_list()
        date2codelist[date] = tmp
    code_set = set([])
    code2gics_info = get_gics_code2gics_info()
    name2code = get_name2code(hs300_wgt_data)
    code2name = dict([(val, key) for key, val in name2code.items()])
    opt_df = pd.DataFrame()
    for date in opt_date_list:
        if date < 20050429:
            continue
        nearly_date_index = bisect.bisect_right(date_list, date)
        nearly_date = date_list[nearly_date_index - 1]
        day_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == nearly_date]
        day_code_list = day_wgt_data['ts_code'].to_list()
        specific_day_daily_data = daily_data[daily_data.trade_date == date]
        wgt = compute_weight_by_cirv_market_value(specific_day_daily_data, day_code_list)
        code_set.update(set(day_code_list))
        nan_col_code = list(code_set)
        nan_col_name, nan_col_industry_name, nan_col_gics_code, nan_col_weight = [],[],[],[]
        for i in tqdm(nan_col_code):
            nan_col_name.append(code2name[i])
            nan_col_industry_name.append(code2gics_info[i][0])
            nan_col_gics_code.append(code2gics_info[i][1])
            tmp_weight = wgt[i] if i in wgt else 0
            nan_col_weight.append(tmp_weight)
        nan_dict = {
            'trade_date': date,
            'ts_code': nan_col_code,
            'stock_name': nan_col_name,
            'gics_code': nan_col_gics_code,
            'industry_name': nan_col_industry_name,
            'weight': nan_col_weight,
        }
        nan_df = pd.DataFrame(nan_dict)
        opt_df = opt_df.append(nan_df, ignore_index=True)
    opt_df.to_csv(fixed_opt_weight_data_path, index=False)
    return opt_df


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


def compute_weight_by_cirv_market_value(daily_data, weight_code):
    ts_code = daily_data['ts_code'].to_numpy()
    circ_mv = daily_data['circ_mv'].to_numpy()
    total_mv = daily_data['total_mv'].to_numpy()
    weight = circ_mv / total_mv
    weight = myround(weight)
    correct_mv = total_mv * weight
    round_market_value = zip(ts_code, correct_mv)
    round_market_value_dict = dict([(k, v) for k, v in round_market_value if k in weight_code])
    market_value_sum = sum(round_market_value_dict.values())
    wgt = dict([(k, 100*v/market_value_sum) for k, v in round_market_value_dict.items()])
    return wgt


def main():
    start = time()
    month_const = config.month_const
    lag_num = config.lag_num
    vol_lag = config.vol_lag
    rsi_lag = config.rsi_lag
    daily_data_path = config.daily_data_path
    balance_data_path = config.balance_data_path
    income_data_path = config.income_data_path
    cashflow_data_path = config.cashflow_data_path
    fina_indicator_data_path = config.fina_indicator_data_path
    opt_monthly_data_path = config.opt_monthly_data_path
    trade_monthly_data_path = config.trade_monthly_data_path
    weight_data_path = config.weight_data_path
    all_data_path = config.all_data_path
    opt_monthly_price_change_data_path = config.opt_monthly_price_change_data_path
    trade_monthly_price_change_data_path = config.trade_monthly_price_change_data_path

    daily_data = read_csv(daily_data_path, '\t')
    daily_data = daily_data.sort_values(['ts_code', 'trade_date'])
    daily_data = prepare_fixed_daily_data(daily_data)
    weight_data = read_csv(weight_data_path)
    # opt_date_list = weight_data['trade_date'].drop_duplicates().to_list()
    opt_date_list = prepare_week_trade_date_list(daily_data)
    opt_date_list.sort()
    code_list = weight_data['ts_code'].drop_duplicates().to_list()
    # trade_date_list, opt2trade = prepare_one_day_late_trade_date_list(daily_data, opt_date_list)
    trade_date_list = opt_date_list
    opt_monthly_data, trade_monthly_data = prepare_fixed_monthly_data_df(opt_date_list, trade_date_list, code_list)
    opt_monthly_data, trade_monthly_data = norm_df(opt_monthly_data), norm_df(trade_monthly_data)
    opt_monthly_trading_data = merge_data(opt_monthly_data, daily_data)
    trade_monthly_trading_data = merge_data(trade_monthly_data, daily_data)
    opt_monthly_chg_data = prepare_monthly_price_change_data(opt_monthly_trading_data,
                                                             month_const, lag_num,
                                                             opt_monthly_price_change_data_path)
    trade_monthly_chg_data = prepare_monthly_price_change_data(trade_monthly_trading_data,
                                                               month_const, lag_num,
                                                               trade_monthly_price_change_data_path)
    opt_monthly_trading_data = merge_data(opt_monthly_trading_data, opt_monthly_chg_data)
    trade_monthly_trading_data = merge_data(trade_monthly_trading_data, trade_monthly_chg_data)
    opt_monthly_trading_data["12monPC_1monPC"] = opt_monthly_trading_data["last_12mon_pricechange_lag"] - \
                                                 opt_monthly_trading_data["last_1mon_pricechange"]
    trade_monthly_trading_data["12monPC_1monPC"] = trade_monthly_trading_data["last_12mon_pricechange_lag"] - \
                                                 trade_monthly_trading_data["last_1mon_pricechange"]
    opt_monthly_trading_data.to_csv(opt_monthly_data_path, index=False)
    trade_monthly_trading_data.to_csv(trade_monthly_data_path, index=False)

    balance_data = read_csv(balance_data_path, sep='\t')
    income_data = read_csv(income_data_path, sep='\t')
    cash_flow_data = read_csv(cashflow_data_path, sep='\t')
    financial_indicator_data = read_csv(fina_indicator_data_path, sep='\t')
    vol_data = prepare_vol_data_df(daily_data, vol_lag)
    rsi_data = prepare_rsi_data_df(daily_data, rsi_lag)
    financial_sheet = prepare_full_financial_sheet(balance_data, income_data,
                                                   cash_flow_data, daily_data, config.financial_blank_df)
    financial_indicator_data = prepare_indicator_sheet(financial_indicator_data,
                                                       daily_data, config.financial_indicator_blank_df)
    weight = read_csv(weight_data_path)
    liab_and_mv = daily_data[['ts_code', 'trade_date', 'total_mv', 'pb', 'close']]
    # fix_wgt_data = prepare_fix_wgt_data_df(weight)
    fix_wgt_data = prepare_week_fix_wgt_data_df(weight, opt_date_list, daily_data)
    all_data = merge_data(fix_wgt_data, vol_data)
    all_data = merge_data(all_data, rsi_data)
    all_data = merge_data(all_data, financial_sheet)
    all_data = merge_data(all_data, financial_indicator_data)
    all_data = merge_data(all_data, liab_and_mv)
    all_data["Log_mkt_Cap"] = np.log(list(all_data["total_mv"]))
    all_data["Cash_Over_MktCap"] = all_data.fcff.div(all_data.total_mv, axis=0)
    all_data["Rev_Over_mktCap"] = all_data.revenue.div(all_data.total_mv, axis=0)
    all_data["NOCF_Over_Debt"] = all_data.n_cashflow_act.div(all_data.total_liab, axis=0)
    all_data.to_csv(all_data_path, index=False)
    print('cost time:', time()-start)


if __name__ == '__main__':
    main()
