import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import copy
import statsmodels.api as sm
from data_process import *
import bisect
from collections import namedtuple

trim_perct = 5  # the cutoff percent for the percent winsorizing

std_num = 3  # the cutoff number of the standard deviation for winsorzing

listed_num = 1  # only choose the industry has at least 5 listed firms

factor_names = ["Size_Factor", 'Volatility_Factor', 'IdioVolatility_Factor', 'RSI_Factor', 'Momentum_Factor',
                'Quality_Factor', 'Value_Factor']  # the names of the five factors

Overall_weight = [0.15, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]  # the weight of five factors in the overall factor: factor_names

Quality_weight = [0.25, 0.25, 0.5]  # the weight for the quality factor: Rev_Over_mktCap, Accural, NOCF_Over_Debt

Value_weight = [1 / 3, 1 / 3, 1 / 3]  # the weight for the value factor: Cash_Over_MktCap, Rev_Over_mktCap, BP

vol_lag = 60  # the lag for the volatility

rsi_lag = 10  # the lag for the RSI

balance_data_path = r'./data/399300.SZ_balancesheetdata.txt'
income_data_path = r'./data/399300.SZ_incomedata.txt'
cashflow_data_path = r'./data/399300.SZ_cashflowdata.txt'
fina_indicator_data_path = r'./data/399300.SZ_fina_indicatordata.txt'
hs300_trading_monthly_fill_nan_path = r'./data/tmp/HS300tradingmonthly_fill_nan.txt'


def prepare_balance_sheet_dict(specific_code_balance_sheet_data):
    f_ann_date = specific_code_balance_sheet_data['f_ann_date']
    ann_date = specific_code_balance_sheet_data['ann_date']
    end_date = specific_code_balance_sheet_data['end_date']
    total_liab = specific_code_balance_sheet_data['total_liab']
    balance_sheet_dict = dict([(x, (y, z, w)) for x, y, z, w in
                               zip(f_ann_date, ann_date, end_date, total_liab)])
    return balance_sheet_dict


def get_code_list(hs300_all_data):
    code_list = hs300_all_data['ts_code'].drop_duplicates().to_list()
    return code_list


def merge_balance_sheet(hs300_all_data, hs300_balance_sheet):
    code_list = get_code_list(hs300_all_data)
    new_hs300_all_data = pd.DataFrame()
    tmp_date_list = hs300_all_data['trade_date'].drop_duplicates().to_list()
    tmp_date_list.sort()
    for code in code_list:
        code_hs300_all_data = hs300_all_data[hs300_all_data.ts_code == code].reset_index(drop=True)
        code_hs300_all_data['total_liab'] = np.nan
        all_data_date_list = code_hs300_all_data['trade_date'].drop_duplicates().to_list()
        all_data_date_list.sort()
        code_hs300_all_data = code_hs300_all_data.set_index(keys='trade_date')
        code_hs300_balance_data = hs300_balance_sheet[hs300_balance_sheet.ts_code == code]
        balance_data_dict = prepare_balance_sheet_dict(code_hs300_balance_data)
        balance_date_list = list(balance_data_dict.keys())
        balance_date_list.sort()
        for date in all_data_date_list:
            b_date_index = bisect.bisect_left(balance_date_list, date)
            b_date = balance_date_list[b_date_index - 1]
            tmp = balance_data_dict[b_date][-1]
            while np.isnan(balance_data_dict[b_date][-1]):
                b_date_index -= 1
                b_date = balance_date_list[b_date_index - 1]
            code_hs300_all_data[date, 'total_liab'] = balance_data_dict[b_date][-1]
        new_hs300_all_data.append(code_hs300_all_data.reset_index())
    new_hs300_all_data.to_csv('./data/tmp/hs300_add_balance.csv',index=False)
    return new_hs300_all_data


if __name__ == '__main__':
    HS300_tradingdata = read_csv(hs300_trade_data_path, '\t')
    hs300_trading_monthly_fill_nan = pd.read_csv(hs300_trading_monthly_fill_nan_path)
    HS300alldata = hs300_trading_monthly_fill_nan

    HS300_balancesheetdata = read_csv(balance_data_path, sep='\t')
    HS300_incomedata = read_csv(income_data_path, sep='\t')
    HS300_cashflowdata = read_csv(cashflow_data_path, sep='\t')
    HS300_fina_indicatordata = read_csv(fina_indicator_data_path, sep='\t')
    merge_balance_sheet(HS300alldata, HS300_balancesheetdata)

#
# ## 2.5 append the financial data into HS300alldata
# HS300alldata['total_liab'] = np.nan
# HS300alldata['n_cashflow_act'] = np.nan
# HS300alldata['revenue'] = np.nan
# HS300alldata['fcff'] = np.nan
# HS300alldata['roa'] = np.nan
# HS300alldata['q_opincome'] = np.nan
#
# for i in tqdm(range(HS300alldata.shape[0])):
#     mbalancesheet = HS300_balancesheetdata.loc[HS300_balancesheetdata.ts_code == HS300alldata.loc[i, 'ts_code']]
#     mbalancesheet_last = mbalancesheet.loc[mbalancesheet.f_ann_date <= HS300alldata.loc[i, 'trade_date']].reset_index(
#         drop=True)
#     if mbalancesheet_last.shape[0] > 0:
#         m_total_liab = mbalancesheet_last.loc[~(np.isnan(mbalancesheet_last.total_liab))].reset_index(drop=True)
#         if m_total_liab.shape[0] > 0:
#             HS300alldata.loc[i, 'total_liab'] = m_total_liab.loc[0, 'total_liab']
#     mcashflow = copy.deepcopy(HS300_cashflowdata.loc[HS300_cashflowdata.ts_code == HS300alldata.loc[i, 'ts_code']])
#     mcashflow_last = copy.deepcopy(
#         mcashflow.loc[mcashflow.f_ann_date <= HS300alldata.loc[i, 'trade_date']].reset_index(drop=True))
#     if mcashflow_last.shape[0] > 0:
#         m_n_cashflow_act = copy.deepcopy(
#             mcashflow_last.loc[~(np.isnan(mcashflow_last.n_cashflow_act))].reset_index(drop=True))
#         if m_n_cashflow_act.shape[0] > 0:
#             HS300alldata.loc[i, 'n_cashflow_act'] = m_n_cashflow_act.loc[0, 'n_cashflow_act']
#     ## income: revenue
#     mincome = copy.deepcopy(HS300_incomedata.loc[HS300_incomedata.ts_code == HS300alldata.loc[i, 'ts_code']])
#     mincome_last = copy.deepcopy(
#         mincome.loc[mincome.f_ann_date <= HS300alldata.loc[i, 'trade_date']].reset_index(drop=True))
#     if mincome_last.shape[0] > 0:
#         m_income = copy.deepcopy(mincome_last.loc[~(np.isnan(mincome_last.revenue))].reset_index(drop=True))
#         if m_income.shape[0] > 0:
#             HS300alldata.loc[i, 'revenue'] = m_income.loc[0, 'revenue']
#     ## financial indicator:fcff, roa and q_opincome
#     mindicator = copy.deepcopy(
#         HS300_fina_indicatordata.loc[HS300_fina_indicatordata.ts_code == HS300alldata.loc[i, 'ts_code']])
#     mindicator_last = copy.deepcopy(
#         mindicator.loc[mindicator.ann_date <= HS300alldata.loc[i, 'trade_date']].reset_index(drop=True))
#     if mindicator_last.shape[0] > 0:
#         m_fcff = copy.deepcopy(mindicator_last.loc[~(np.isnan(mindicator_last.fcff))].reset_index(drop=True))
#         if m_fcff.shape[0] > 0:
#             HS300alldata.loc[i, 'fcff'] = m_fcff.loc[0, 'fcff']
#         m_roa = copy.deepcopy(mindicator_last.loc[~(np.isnan(mindicator_last.roa))].reset_index(drop=True))
#         if m_roa.shape[0] > 0:
#             HS300alldata.loc[i, 'roa'] = m_roa.loc[0, 'roa']
#         m_q_opincome = copy.deepcopy(
#             mindicator_last.loc[~(np.isnan(mindicator_last.q_opincome))].reset_index(drop=True))
#         if m_q_opincome.shape[0] > 0:
#             HS300alldata.loc[i, 'q_opincome'] = m_q_opincome.loc[0, 'q_opincome']
#
#
# HS300alldata.to_csv('./data/tmp/hs300_all_data.csv')
# HS300alldata["Log_mkt_Cap"] = np.log(list(HS300alldata["total_mv"]))
# HS300alldata["Cash_Over_MktCap"] = HS300alldata.fcff.div(HS300alldata.total_mv, axis=0)
# HS300alldata["Rev_Over_mktCap"] = HS300alldata.revenue.div(HS300alldata.total_mv, axis=0)
# HS300alldata["NOCF_Over_Debt"] = HS300alldata.n_cashflow_act.div(HS300alldata.total_liab, axis=0)
#
# ################################################################
# ############ Step Three: volatility for the last 60 trading days
# HS300alldata['Volatility'] = np.nan
# for i in tqdm(range(HS300alldata.shape[0])):
#     mtradingdata_vol = copy.deepcopy(HS300_tradingdata.loc[HS300_tradingdata.ts_code == HS300alldata.loc[i, 'ts_code']])
#     mtradingdata_vol1 = copy.deepcopy(
#         mtradingdata_vol.loc[mtradingdata_vol.trade_date < HS300alldata.loc[i, 'trade_date']].reset_index(drop=True))
#     ## volatility
#     if (mtradingdata_vol1.shape[0] >= vol_lag):
#         HS300alldata.loc[i, 'Volatility'] = np.nanstd(mtradingdata_vol1.loc[:(vol_lag - 1), 'pct_chg_hfq'])
#         # print(str(i)+' : ' + 'has volatility data')
#     else:
#         print(str(i) + ' : ' + 'has no volatility data')
# HS300alldata.to_csv('./data/tmp/hs300_all_data.csv')
# ############################################################################
# ############ Step Four: Idiosyncratic volatility and Relative Strength Index
# ## input the daily trading data of HS300 index
# index_data_path = r'./data/HS300indexdata.txt'
# HS300indexdata = pd.read_csv(index_data_path, sep='\t')
# HS300alldata['Idio_vol'] = np.nan
# HS300alldata['RSI'] = np.nan
# for i in tqdm(range(HS300alldata.shape[0])):
#     mtradingdata_ivol = copy.deepcopy(
#         HS300_tradingdata.loc[HS300_tradingdata.ts_code == HS300alldata.loc[i, 'ts_code']])
#     mtradingdata_ivol1 = copy.deepcopy(
#         mtradingdata_ivol.loc[mtradingdata_ivol.trade_date < HS300alldata.loc[i, 'trade_date']].reset_index(drop=True))
#     ## Idiosyncratic volatility for the last 60 days
#     if (mtradingdata_ivol1.shape[0] >= vol_lag):
#         mtradingdata_ivol1_last = copy.deepcopy(mtradingdata_ivol1.loc[:(vol_lag - 1)].reset_index(drop=True))
#         mtradingdata_ivol1_last = copy.deepcopy(
#             mtradingdata_ivol1_last.merge(HS300indexdata, how='inner', left_on=['trade_date'], right_on=["trade_date"]))
#         y = mtradingdata_ivol1_last.pct_chg_hfq
#         X = mtradingdata_ivol1_last.pct_chg_y
#         X = sm.add_constant(X)
#         est = sm.OLS(y, X).fit()
#         y_hat = est.predict(X)
#         HS300alldata.loc[i, 'Idio_vol'] = np.nanstd(y - y_hat)
#     else:
#         print(str(i) + ' : ' + 'has no idiosyncratic volatility data')
#     if (mtradingdata_ivol1.shape[0] >= rsi_lag):
#         mtradingdata_rsi_last = copy.deepcopy(mtradingdata_ivol1.loc[:(rsi_lag - 1)].reset_index(drop=True))
#         mGain = sum(mtradingdata_rsi_last.loc[mtradingdata_rsi_last.pct_chg_hfq > 0, 'pct_chg_hfq'])
#         mLoss = np.abs(sum(mtradingdata_rsi_last.loc[mtradingdata_rsi_last.pct_chg_hfq < 0, 'pct_chg_hfq']))
#         HS300alldata.loc[i, 'RSI'] = (mGain / (mGain + mLoss)) * 100
# HS300alldata.to_csv('./data/tmp/hs300_all_data.csv')
# exit()










