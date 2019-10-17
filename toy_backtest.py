import gurobipy
import pandas as pd
import numpy as np
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import time
industry_upper_bound = 0.01
industry_lower_bound = 0.01

factor_upper_bound = 0.35
factor_lower_bound = 0.00

stock_upper_bound = 0.01
stock_lower_bound = 0.01

constr_factor = ['Size_Factor', 'IdioVolatility_Factor', 'RSI_Factor', 'Quality_Factor', 'Value_Factor']


def data_process(hs300_trading_monthly_path, raw_factor_path):
    hs300_trading_monthly = pd.read_csv(hs300_trading_monthly_path, sep=',')
    raw_factor = pd.read_csv(raw_factor_path, sep=',')
    date_list = raw_factor['Date'].drop_duplicates()
    date_list = date_list.sort_values()
    date_list = date_list.to_list()
    return hs300_trading_monthly, raw_factor, date_list


def init_ans_df(date_list):
    mrawret = pd.DataFrame({'Date': date_list})
    mrawret['FactorModel'] = np.nan
    mrawret['HS300Index'] = np.nan
    mrawret['NetRet'] = np.nan
    mrawret.loc[0, ['FactorModel', 'HS300Index', 'NetRet']] = 0

    mrawret['FactorModel_cum'] = np.nan
    mrawret['HS300Index_cum'] = np.nan
    mrawret['NetRet_cum'] = np.nan
    mrawret.loc[0, ['FactorModel_cum', 'HS300Index_cum', 'NetRet_cum']] = 1000
    return mrawret


def init_model(weigth_num, original_weight, overall_score):
    model = gurobipy.Model('factor_model')
    weight = {}
    for j in range(weigth_num):
        lb = max(0, original_weight[j] - stock_lower_bound)
        ub = min(1, original_weight[j] + stock_upper_bound)
        weight[j] = model.addVar(lb=lb, ub=ub, name='weight_' + str(j), vtype=gurobipy.GRB.SEMICONT)
    weight = gurobipy.tupledict(weight)
    weight_binary = model.addVars(weigth_num, lb=0, ub=1,
                                  vtype=gurobipy.GRB.BINARY, name='weight_binary')
    model.setObjective(weight.prod(overall_score), gurobipy.GRB.MAXIMIZE)
    return model, weight, weight_binary


def add_factor_constr(model, weight, factor_data, original_weight,factor_name):
    factor_score = factor_data[factor_name].to_list()
    factor_weigth_score = np.dot(original_weight, factor_score)
    upper_bound = factor_weigth_score * (1 + factor_upper_bound)
    lower_bound = factor_weigth_score * (1 + factor_lower_bound)
    model.addConstr(weight.prod(factor_score) >= lower_bound)
    model.addConstr(weight.prod(factor_score) <= upper_bound)


def add_stock_constr(model, weight, weight_binary, weight_num):
    # 选股数量约束，要求 [0，60]
    model.addConstrs(weight[j] <= weight_binary[j] for j in range(weight_num))
    model.addConstr(gurobipy.quicksum(weight_binary) <= 100, name='stock num')
    # 权重和约束，weight和为 1
    model.addConstr(gurobipy.quicksum(weight) == 1, 'budge')


def add_industry_constr(model,weight,factor_data):
    all_industry = factor_data['Gics'].drop_duplicates()
    all_industry = all_industry.to_list()
    all_industry.sort()
    for industry in all_industry:
        specific_industry_df = factor_data[factor_data.Gics == industry]
        if len(specific_industry_df) <= 5:
            # print('industry {}  be skipped'.format(industry))
            # print('industry has length {}'.format(len(specific_industry_df)))
            continue
        model.addConstr(sum(weight[j] for j in specific_industry_df.index) <=
                        sum(specific_industry_df.weight) / 100 + industry_upper_bound,
                        name='industry_upper_bound: ' + industry)
        model.addConstr(sum(weight[j] for j in specific_industry_df.index) >=
                        sum(specific_industry_df.weight) / 100 - industry_lower_bound,
                        name='industry_lowwer_bound: ' + industry)


def main():
    """超参"""
    hs300_trading_monthly_path = r'./data/HS300tradingmonthly.txt'
    raw_factor_path = r'./data/raw_factor_test.csv'
    hs300_trading_monthly, raw_factor, date_list = data_process(hs300_trading_monthly_path, raw_factor_path)
    mrawret = init_ans_df(date_list)
    for idx, date in tqdm(enumerate(date_list)):
        if idx >= len(date_list)-1:
            break
        factor_data = raw_factor[raw_factor.Date == date].reset_index(drop=True)
        original_weight = factor_data.weight.to_numpy()/100
        overall_score = factor_data['Overall_Factor'].to_list()
        weight_num = len(overall_score)
        # optimize
        model, weight, weight_binary = init_model(weight_num,original_weight,overall_score)
        # stock constr
        add_stock_constr(model, weight, weight_binary, weight_num)
        # industry constr
        add_industry_constr(model, weight, factor_data)
        # size factor constr
        for fac in constr_factor:
            add_factor_constr(model, weight, factor_data, original_weight, fac)

        model.optimize()
        # model.computeIIS()
        # model.write("model.ilp")
        vars_opt = pd.DataFrame()
        mcount = 0
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            for v in model.getVars():
                vars_opt.loc[mcount, 'varname'] = v.varname
                vars_opt.loc[mcount, 'value'] = v.x
                mcount += 1
        else:
            print('data {} infeasible'.format(date))
            exit()
            continue
        vars_opt_all_weight = vars_opt['value']
        vars_opt_all_weight = vars_opt_all_weight.to_numpy()
        vars_opt_weight = vars_opt_all_weight[:weight_num]
        next_date = date_list[idx+1]
        hs300_data_monthly = hs300_trading_monthly[hs300_trading_monthly.Date == next_date].reset_index(drop=True)
        factor_data = factor_data.merge(hs300_data_monthly,
                                        how='left',
                                        left_on=['Codes'],
                                        right_on=['Codes'])\

        mrawret.loc[idx+1, 'FactorModel'] = np.dot(vars_opt_weight, factor_data['1mon_tradingreturn'])
        mrawret.loc[idx+1, 'HS300Index'] = np.dot(original_weight, factor_data['1mon_tradingreturn'])
        mrawret.loc[idx+1, 'NetRet'] = mrawret.loc[idx+1, 'FactorModel'] - mrawret.loc[idx+1, 'HS300Index']

        mrawret.loc[idx+1, 'FactorModel_cum'] = mrawret.loc[idx, 'FactorModel_cum'] * (1 + mrawret.loc[idx+1, 'FactorModel'])
        mrawret.loc[idx+1, 'HS300Index_cum'] = mrawret.loc[idx, 'HS300Index_cum'] * (1 + mrawret.loc[idx+1, 'HS300Index'])
        mrawret.loc[idx+1, 'NetRet_cum'] = mrawret.loc[idx, 'NetRet_cum'] * (1 + mrawret.loc[idx+1, 'NetRet'])

    print(sum(mrawret.loc[1:, 'NetRet'] >= 0)/len(mrawret.loc[1:, 'NetRet']))
    print(np.mean(mrawret.loc[1:, 'NetRet']) * 12)

    mDate_ret = [str(mrawret.loc[x, 'Date']) for x in range(mrawret.shape[0])]
    mrawret['mrawret'] = mDate_ret
    mrawret['mrawret'] = pd.to_datetime(mrawret['mrawret'])

    fig, ax = plt.subplots()
    ax.plot(mrawret['mrawret'], mrawret.FactorModel_cum, color="red", linewidth=1.25, label="Factor Model")
    ax.plot(mrawret['mrawret'], mrawret.HS300Index_cum, color="blue", linewidth=1.25, label="HS300Index Strategy")
    ax.plot(mrawret['mrawret'], mrawret.NetRet_cum, color="green", linewidth=1.25, label="Net Return")

    ax.legend(loc=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('The Cumulative Return for the Factor Model and the HS300 Strategy')
    plt.show()
    fig.savefig("./Factor_Model_Payoff_plot_Wenzhou.pdf", dpi=100)


def data_explore():
    hs300_trading_monthly_path = r'./data/HS300tradingmonthly.txt'
    raw_factor_path = r'./data/raw_factor_test.csv'
    hs_300trading_monthly, raw_factor, date_list = data_process(hs300_trading_monthly_path, raw_factor_path)
    codes_dict = {}
    codes_set = set([])
    for idx, date in enumerate(date_list):
        factor_date = raw_factor[raw_factor.Date == date]
        factor_codes = factor_date['Codes']
        codes_dict[date] = len(factor_codes)
        codes_set.update(factor_codes.to_list())
    pprint(codes_dict)
    print(len(codes_set))


if __name__ == '__main__':
    main()
