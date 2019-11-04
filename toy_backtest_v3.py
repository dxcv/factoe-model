import numpy as np
import gurobipy
import copy
from matplotlib import pyplot as plt
from tools import *


hs300_trading_monthly_path = r'./data/HS300tradingmonthly.txt'
raw_factor_path = r'./data/raw_factor_test_v1.csv'
hs300_trade_data_path = r'./data/399300.SZ_tradingdata.txt'
hs300_wgt_path = r'./data/HS300_idx_wt.csv'
hs300_idx_codes_path = r'./data/399300.SZ_weight.txt'
hs300_all_data_path = r'./data/HS300alldata_vol2.txt'

industry_upper_bound = 0.01
industry_lower_bound = 0.01

factor_upper_bound = 0.35
factor_lower_bound = 0.00

stock_upper_bound = 0.005
stock_lower_bound = 0.005

constr_factor = ['size_factor', 'RSI_factor', 'quality_factor', 'value_factor']


def data_process(hs300_trading_monthly_path, raw_factor_path):
    hs300_trading_monthly = pd.read_csv(hs300_trading_monthly_path, sep=',')
    raw_factor = pd.read_csv(raw_factor_path, sep=',')
    date_list = raw_factor['Date'].drop_duplicates()
    date_list = date_list.sort_values()
    date_list = date_list.to_list()
    return hs300_trading_monthly, raw_factor, date_list


def norm_df(hs300_trading_monthly,hs300_trade_data, hs300_wgt_data, raw_factor):
    # hs300_trade_data = hs300_trade_data.rename(columns={"Date": "trade_date", "Codes": "ts_code"})
    raw_factor = raw_factor.replace({'Codes': '000022.SZ'}, '001872.SZ')
    hs300_trading_monthly = hs300_trading_monthly.replace({'ts_code': '000022.SZ'}, '001872.SZ')
    hs300_trade_data = hs300_trade_data.replace({'ts_code': '000022.SZ'}, '001872.SZ')
    hs300_wgt_data = hs300_wgt_data.rename(columns={"Date": "trade_date", "sec_code": "ts_code"})
    raw_factor = raw_factor.rename(columns={"Date": "trade_date", "Codes": "ts_code"})
    raw_factor = raw_factor.loc[:, ~raw_factor.columns.duplicated()]
    return hs300_trading_monthly, hs300_trade_data, hs300_wgt_data, raw_factor


def weight_analysis(hs300_wgt_data, date_list):
    ans = pd.DataFrame(index=date_list)
    for date in date_list:
        tmp = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        tmp = tmp['weight']
        tmp = sum(tmp)
        ans.loc[date, 'sum'] = tmp/100
    print(ans.describe())
    exit()


def get_hs300_stock_code(hs300_wgt_data, date):
    return hs300_wgt_data[hs300_wgt_data.trade_date == date]['ts_code'].to_list()


def update_opt_weight_key(hs300_code, weight_set):
    weight_set.update(set(hs300_code))
    return weight_set


def get_stock_close_price(trade_data, date=None):
    if date:
        trade_data = trade_data[trade_data.trade_date == date]
    close_price = {}
    num = len(trade_data)
    for i in range(num):
        tmp = trade_data.iloc[i]
        ts_code = tmp['ts_code']
        price = tmp['close']
        close_price[ts_code] = price
    return close_price


def get_ts_code_original_weight(hs300_wgt_data, date=None):
    if date:
        hs300_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == date]
    original_weight = {}
    num = len(hs300_wgt_data)
    date_list = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
    assert len(date_list) == 1, print('the hs300_wgt_data has more one days data'.format(date_list))
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


def get_factor_score(factor_data, factor_name):
    df = factor_data[['ts_code', factor_name]]
    factor_score = {}
    for i in range(len(df)):
        tmp = df.iloc[i]
        code = tmp['ts_code']
        score = tmp[factor_name]
        factor_score[code] = score
    return factor_score


def get_position(wgt_opt_df, date=None):
    if date:
        wgt_opt_df = wgt_opt_df[wgt_opt_df.trade_date == date]
    num = len(wgt_opt_df)
    position = {}
    wgt_opt_df_columns = wgt_opt_df.columns
    assert 'ts_code' in wgt_opt_df_columns, print(wgt_opt_df_columns)
    assert 'position'in wgt_opt_df_columns, print(wgt_opt_df_columns)
    for i in range(num):
        tmp = wgt_opt_df.iloc[i]
        position[tmp['ts_code']] = tmp['position']
    return position


def get_adjust_weight(wgt_opt_df, trade_data, date):
    df = wgt_opt_df[wgt_opt_df.trade_date == date]
    trade_data= trade_data[trade_data.trade_date == date]
    close_price = get_stock_close_price(trade_data)
    position = get_position(df)
    adjust_weight = compute_weight_by_position(position, close_price)
    return adjust_weight


def compute_position(close_price, weight):
    position = {}
    # assert close_price.keys() == weight.keys()
    for key, val in weight.items():
        position[key] = val / close_price[key]
    return position


def compute_weight_by_position(position, close_price):
    weight = {}
    # symbol = '000063.SZ'
    all_position_price = dict_prod(position, close_price)
    for key, val in position.items():
        # if key == symbol:
        #     print('position val :', val, 'close price :', close_price[key], 'all_position_price', all_position_price)
        weight[key] = (val * close_price[key])/all_position_price
    assert sum(weight.values()) >= 0.9999999999, print(sum(weight.values()))
    return weight


def init_wgt_opt_df(hs300_wgt_data):
    df = copy.deepcopy(hs300_wgt_data)
    return df


def init_ans_df(date_list):
    mrawret = pd.DataFrame({'trade_date': date_list})
    mrawret['factor_model'] = np.nan
    mrawret['hs300index'] = np.nan
    mrawret['net_ret'] = np.nan
    mrawret.loc[0, ['factor_model', 'hs300index', 'net_ret']] = 0

    mrawret['factor_model_cum'] = np.nan
    mrawret['hs300index_cum'] = np.nan
    mrawret['net_ret_cum'] = np.nan
    mrawret.loc[0, ['factor_model_cum', 'hs300index_cum', 'net_ret_cum']] = 1000
    return mrawret


def init_model(hs300_ts_code, original_weight, overall_score, adjust_weight, suspend_stock):
    model = gurobipy.Model('factor_model')
    weight = {}
    weight_binary = {}
    for code in hs300_ts_code:
        lb = max(0, original_weight[code] - stock_lower_bound)
        ub = min(1, original_weight[code] + stock_upper_bound)
        if adjust_weight and (code in adjust_weight) and (code in suspend_stock):
            if adjust_weight[code] != 0:
                lb = min(lb, adjust_weight[code])
            ub = max(ub, adjust_weight[code])
        weight[code] = model.addVar(lb=lb, ub=ub, name='optweight_' + code, vtype=gurobipy.GRB.SEMICONT)
        weight_binary[code] = model.addVar(lb=0, ub=1, vtype=gurobipy.GRB.BINARY, name='biweight_' + code)
    weight = gurobipy.tupledict(weight)
    weight_binary = gurobipy.tupledict(weight_binary)
    model.setObjective(weight.prod(overall_score), gurobipy.GRB.MAXIMIZE)
    return model, weight, weight_binary


def add_trade_constr(model, weight, weight_binary,
                     suspend_stock, adjust_weight):
    if not adjust_weight:
        adjust_weight = {}
        for stock in suspend_stock:
            adjust_weight[stock] = 0
    for stock in suspend_stock:
        if stock in adjust_weight and adjust_weight[stock] != 0:
            model.addConstr(weight[stock] == adjust_weight[stock], name='trade_cons_'+stock)


def add_stock_constr(model, weight, weight_binary, hs300_ts_code):
    # 选股数量约束，要求 [0，60]
    model.addConstrs(weight[j] <= weight_binary[j] for j in hs300_ts_code)
    model.addConstr(gurobipy.quicksum(weight_binary) <= 100, name='stock num')
    # 权重和约束，weight和为 1
    model.addConstr(gurobipy.quicksum(weight) == 1, 'budge')


def add_industry_constr(model, weight, factor_data, suspend_stock_code, adjust_weight):
    all_industry = factor_data['Gics'].drop_duplicates()
    all_industry = all_industry.to_list()
    all_industry.sort()
    for industry in all_industry:
        specific_industry_df = factor_data[factor_data.Gics == industry]
        if len(specific_industry_df) <= 5:
            continue
        codes = specific_industry_df.ts_code.to_list()
        lb = sum(specific_industry_df.weight) / 100 - industry_lower_bound
        ub = sum(specific_industry_df.weight) / 100 + industry_upper_bound
        opt_industry_weight = sum([weight[j] for j in codes])
        model.addConstr(opt_industry_weight <= ub, name='industry_upper_bound: ' + industry)
        model.addConstr(opt_industry_weight >= lb, name='industry_lowwer_bound: ' + industry)


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
    hs300_trading_monthly, hs300_trade_data, hs300_wgt_data, raw_factor = norm_df(hs300_trading_monthly, hs300_trade_data,
                                                                                  hs300_wgt_data, raw_factor)
    wgt_opt_df = pd.DataFrame()
    # weight_analysis(hs300_wgt_data, date_list)
    mrawret = init_ans_df(date_list)
    pre_date = None
    track = []
    symbol = '000063.SZ'
    for idx, date in enumerate(date_list):
        print('now date is {}'.format(date))
        if idx >= len(date_list)-1:
            break
        factor_data = raw_factor[raw_factor.trade_date == date]
        overall_score = get_factor_score(factor_data, 'overall_factor')
        specific_day_trade_data = hs300_trade_data[hs300_trade_data.trade_date == date]
        specific_day_hs300_wgt = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        trade_ts_codes = specific_day_trade_data['ts_code']
        hs300_ts_codes = specific_day_hs300_wgt['ts_code']
        suspend_stock_code = select_suspend_stock(trade_ts_codes, hs300_ts_codes)
        original_weight = get_ts_code_original_weight(specific_day_hs300_wgt)
        # init model
        if pre_date:
            adjust_weight = get_adjust_weight(wgt_opt_df, raw_factor, pre_date)
        else:
            adjust_weight = None
        model, weight, weight_binary = init_model(hs300_ts_codes, original_weight,
                                                  overall_score, adjust_weight,
                                                  suspend_stock_code)
        add_stock_constr(model, weight, weight_binary, hs300_ts_codes)
        add_trade_constr(model, weight, weight_binary, suspend_stock_code, adjust_weight)
        add_industry_constr(model, weight, factor_data, suspend_stock_code, adjust_weight)
        for fac in constr_factor:
            add_factor_constr(model, weight, factor_data, original_weight, fac)
        model.optimize()

        vars_opt = pd.DataFrame(index=hs300_ts_codes.to_list())
        vars_opt['trade_date'] = date
        opt_weight = {}
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            for v in model.getVars():
                varname = v.varname
                varname = varname.split('_')
                ts_code = varname[-1]
                if ts_code == symbol:
                    track.append(v.x)
                colunm_name = varname[0]
                if colunm_name == 'optweight':
                    opt_weight[ts_code] = v.x
                vars_opt.loc[ts_code, colunm_name] = v.x
                vars_opt.loc[ts_code, 'ts_code'] = ts_code
        else:
            print('data {} infeasible'.format(date))
            model.computeIIS()
            model.write("./ilp/model_{}.ilp".format(date))
            # print('date is {} symbol is {} adjust weight is {}'.format(date, symbol, adjust_weight[symbol]), track)
            exit()
            continue
        close_price = get_stock_close_price(factor_data)
        stock_position = compute_position(close_price, opt_weight)
        track.append(['close', symbol, close_price[symbol]])
        track.append(['position', symbol, stock_position[symbol]])
        for key, val in stock_position.items():
            vars_opt.loc[key, 'position'] = val
        # vars_opt = vars_opt.reset_index()
        if wgt_opt_df.empty:
            wgt_opt_df = vars_opt
        else:
            wgt_opt_df = pd.concat([wgt_opt_df, vars_opt], axis=0)

        next_date = date_list[idx+1]
        pre_date = date
        hs300_data_monthly = hs300_trading_monthly[hs300_trading_monthly.Date == next_date].reset_index(drop=True)
        factor_data = factor_data.merge(hs300_data_monthly, how='left', left_on=['ts_code'], right_on=['ts_code'])
        mon_trade_return = get_factor_score(factor_data, '1mon_tradingreturn')
        mrawret.loc[idx+1, 'factor_model'] = dict_prod(opt_weight, mon_trade_return)
        mrawret.loc[idx+1, 'hs300index'] = dict_prod(original_weight, mon_trade_return)
        mrawret.loc[idx+1, 'net_ret'] = mrawret.loc[idx+1, 'factor_model'] - mrawret.loc[idx+1, 'hs300index']
        mrawret.loc[idx+1, 'factor_model_cum'] = mrawret.loc[idx, 'factor_model_cum'] * \
                                                (1 + mrawret.loc[idx+1, 'factor_model'])
        mrawret.loc[idx+1, 'hs300index_cum'] = mrawret.loc[idx, 'hs300index_cum'] * \
                                               (1 + mrawret.loc[idx+1, 'hs300index'])
        mrawret.loc[idx+1, 'net_ret_cum'] = mrawret.loc[idx, 'net_ret_cum'] * \
                                           (1 + mrawret.loc[idx+1, 'net_ret'])
    wgt_opt_df.to_csv('wgt_opt_df.csv', sep=',')
    print(sum(mrawret.loc[1:, 'net_ret'] >= 0)/len(mrawret.loc[1:, 'net_ret']))
    print(np.mean(mrawret.loc[1:, 'net_ret']) * 12)
    mrawret.to_csv('mrawret.csv', sep=',')
    mDate_ret = [str(mrawret.loc[x, 'trade_date']) for x in range(mrawret.shape[0])]
    mrawret['mrawret'] = mDate_ret
    mrawret['mrawret'] = pd.to_datetime(mrawret['mrawret'])

    fig, ax = plt.subplots()
    ax.plot(mrawret['mrawret'], mrawret.factor_model_cum, color="red", linewidth=1.25, label="Factor Model")
    ax.plot(mrawret['mrawret'], mrawret.hs300index_cum, color="blue", linewidth=1.25, label="HS300Index Strategy")
    ax.plot(mrawret['mrawret'], mrawret.net_ret_cum, color="green", linewidth=1.25, label="Net Return")

    ax.legend(loc=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('The Cumulative Return for the Factor Model and the HS300 Strategy')
    plt.show()
    fig.savefig("./Factor_Model_V1_Payoff_plot_Wenzhou.pdf", dpi=100)


if __name__ == '__main__':
    main()