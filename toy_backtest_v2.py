import pandas as pd
import numpy as np
import gurobipy
import copy
import pickle as pkl
from tools import *
hs300_trading_monthly_path = r'./data/HS300tradingmonthly.txt'
raw_factor_path = r'./data/raw_factor_test.csv'
hs300_trade_data_path = r'./data/399300.SZ_tradingdata_part1.csv'
hs300_wgt_path = r'./data/HS300_idx_wt.csv'
hs300_idx_codes_path = r'./data/399300.SZ_weight.txt'
hs300_all_data_path = r'./data/HS300alldata_vol2.txt'

industry_upper_bound = 0.01
industry_lower_bound = 0.01

factor_upper_bound = 0.35
factor_lower_bound = 0.00

stock_upper_bound = 0.01
stock_lower_bound = 0.01

constr_factor = ['Size_Factor', 'IdioVolatility_Factor', 'RSI_Factor', 'Quality_Factor', 'Value_Factor']


def norm_df(hs300_trade_data,hs300_wgt_data, raw_factor):
    # hs300_trade_data = hs300_trade_data.rename(columns={"Date": "trade_date", "Codes": "ts_code"})
    raw_factor = raw_factor.replace({'Codes': '000022.SZ'}, '001872.SZ')
    hs300_trade_data = hs300_trade_data.replace({'ts_code': '000022.SZ'}, '001872.SZ')
    hs300_wgt_data = hs300_wgt_data.rename(columns={"Date": "trade_date", "sec_code": "ts_code"})
    raw_factor = raw_factor.rename(columns={"Date": "trade_date", "Codes": "ts_code"})
    return hs300_trade_data,hs300_wgt_data,raw_factor


def get_ts_code_original_weight(hs300_wgt_data):
    original_weight = {}
    num = len(hs300_wgt_data)
    date = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
    assert len(date) == 1, print('the hs300_wgt_data has more one days data'.format(date))
    assert num == 300, print('the num of hs300 is not 300 in date {} '.
                             format(date))
    for i in range(num):
        tmp = hs300_wgt_data.iloc[i]
        ts_code = tmp['ts_code']
        weight = tmp['weight']
        assert ts_code is not None, print('check the ts_code in date {} code is {}'.format(date, ts_code))
        assert weight is not None, print('check the weight in date {} code is {}'.format(date, weight))
        original_weight[ts_code] = weight/100
    return original_weight


def get_factor_score(factor_data, factor_name):
    df = factor_data[['ts_code', factor_name]]
    factor_score = {}
    for i in range(len(df)):
        tmp = df.iloc[i]
        code = tmp['ts_code']
        score = tmp[factor_name]
        factor_score[code] = score
    return factor_score


def init_model(hs300_ts_code, original_weight, overall_score):
    model = gurobipy.Model('factor_model')
    weight = {}
    weight_binary = {}
    for code in hs300_ts_code:
        lb = max(0, original_weight[code] - stock_lower_bound)
        ub = min(1, original_weight[code] + stock_upper_bound)
        weight[code] = model.addVar(lb=lb, ub=ub, name='opt-weight_' + code, vtype=gurobipy.GRB.SEMICONT)
        weight_binary[code] = model.addVar(lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='bi-weight_' + code)
    weight = gurobipy.tupledict(weight)
    weight_binary = gurobipy.tupledict(weight_binary)
    model.setObjective(weight.prod(overall_score), gurobipy.GRB.MAXIMIZE)
    return model, weight, weight_binary


def init_wgt_opt_df(hs300_wgt_data):
    df = copy.deepcopy(hs300_wgt_data)
    return df


def get_pre_opt_weight(wgt_opt_df, date):
    df = wgt_opt_df[wgt_opt_df.Date == date]
    num = len(df)
    pre_opt_weight = {}
    for i in range(num):
        tmp = df.iloc[i]
        pre_opt_weight[tmp['ts_code']] = tmp['opt-weight']
    return pre_opt_weight


def add_trade_constr(model, weight, weight_binary,
                     suspend_stock, pre_opt_weight):
    if not pre_opt_weight:
        pre_opt_weight = {}
        for stock in suspend_stock:
            pre_opt_weight[stock] = 0
    for stock in suspend_stock:
        model.addConstr(weight[stock] == pre_opt_weight[stock], name='trade_cons_'+stock)
        # model.addConstr(weight_binary[stock] == 1, name='trade_cons_' + stock)


def add_stock_constr(model, weight, weight_binary, hs300_ts_code):
    # 选股数量约束，要求 [0，60]
    model.addConstrs(weight[j] <= weight_binary[j] for j in hs300_ts_code)
    model.addConstr(gurobipy.quicksum(weight_binary) <= 100, name='stock num')
    # 权重和约束，weight和为 1
    model.addConstr(gurobipy.quicksum(weight) == 1, 'budge')


def add_industry_constr(model, weight, factor_data):
    all_industry = factor_data['Gics'].drop_duplicates()
    all_industry = all_industry.to_list()
    all_industry.sort()
    for industry in all_industry:
        specific_industry_df = factor_data[factor_data.Gics == industry]
        if len(specific_industry_df) <= 5:
            continue
        codes = specific_industry_df.Codes.to_list()
        # if '000022.SZ' in codes:
        #     idx = codes.index('000022.SZ')
        #     codes[idx] = '001872.SZ'
        model.addConstr(sum(weight[j] for j in codes) <= sum(specific_industry_df.weight) / 100 + industry_upper_bound,
                        name='industry_upper_bound: ' + industry)
        model.addConstr(sum(weight[j] for j in codes) >= sum(specific_industry_df.weight) / 100 - industry_lower_bound,
                        name='industry_lowwer_bound: ' + industry)


def add_factor_constr(model, weight, factor_data, original_weight, factor_name):
    factor_score = get_factor_score(factor_data, factor_name)
    factor_weigth_score = dict_prod(original_weight, factor_score)
    upper_bound = factor_weigth_score * (1 + factor_upper_bound)
    lower_bound = factor_weigth_score * (1 + factor_lower_bound)
    model.addConstr(weight.prod(factor_score) >= lower_bound)
    model.addConstr(weight.prod(factor_score) <= upper_bound)


def main():
    hs300_trading_monthly, raw_factor, date_list = data_process(hs300_trading_monthly_path, raw_factor_path)
    hs300_trade_data = pd.read_csv(hs300_trade_data_path, sep='\t')
    hs300_wgt_data = pd.read_csv(hs300_wgt_path, sep=',')
    wgt_opt_df = init_wgt_opt_df(hs300_wgt_data)
    pre_date = None
    hs300_trade_data,hs300_wgt_data,raw_factor = norm_df(hs300_trade_data,hs300_wgt_data,raw_factor)
    # tmp1 = raw_factor['Codes'].drop_duplicates().to_list()
    # tmp2 = hs300_trade_data['ts_code'].drop_duplicates().to_list()
    # ss = hs300_wgt_data['sec_code']
    # print(set(ss).difference(set(tmp1)))
    # print(set(ss).difference(set(tmp2)))
    # exit()
    for idx, date in enumerate(date_list):
        factor_data = raw_factor[raw_factor.trade_date == date]
        overall_score = get_factor_score(factor_data, 'Overall_Factor')
        specific_day_trade_data = hs300_trade_data[hs300_trade_data.trade_date == date]
        specific_day_hs300_wgt = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        trade_ts_codes = specific_day_trade_data['ts_code']
        hs300_ts_codes = specific_day_hs300_wgt['ts_code']
        suspend_stock_code = select_suspend_stock(trade_ts_codes, hs300_ts_codes)
        # init model
        original_weight = get_ts_code_original_weight(specific_day_hs300_wgt)
        model, weight, weight_binary = init_model(hs300_ts_codes, original_weight, overall_score)
        add_stock_constr(model, weight, weight_binary, hs300_ts_codes)
        if pre_date:
            pre_opt_weight = get_pre_opt_weight(wgt_opt_df, pre_date)
        else:
            pre_opt_weight = None
        add_trade_constr(model, weight, weight_binary, suspend_stock_code, pre_opt_weight)
        add_industry_constr(model, weight, factor_data)
        for fac in constr_factor:
            add_factor_constr(model, weight, factor_data, original_weight, fac)
        model.optimize()
        vars_opt = pd.DataFrame(index=hs300_ts_codes)
        vars_opt['trade_date'] = date
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            for v in model.getVars():
                varname = v.varname
                varname = varname.split('_')
                vars_opt.loc[varname[-1], varname[0]] = v.x
                vars_opt.loc[varname[-1], 'ts_code'] = varname[-1]
        else:
            print('data {} infeasible'.format(date))
            exit()
        exit()
        vars_opt['position'] = np.dot(opt_weight, close)
        wgt_opt_df = wgt_opt_df.merge(vars_opt, how='left',
                         left_on=['sec_code', 'Date'], right_on=['ts_code', 'trade_date'])
        print(wgt_opt_df.tail)
        exit()


if __name__ == '__main__':
    main()