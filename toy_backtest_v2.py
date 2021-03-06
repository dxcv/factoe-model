import copy
import numpy as np
from tools import *
from configuration import Config
from BackTest import BackTestHs300
from matplotlib import pyplot as plt
from data_process import read_csv, load_obj


def data_process(opt_monthly_data_path, raw_factor_path, trade_monthly_data_path):
    opt_monthly_data = pd.read_csv(opt_monthly_data_path)
    trade_monthly_data = pd.read_csv(trade_monthly_data_path)
    raw_factor = pd.read_csv(raw_factor_path, sep=',')
    return opt_monthly_data, trade_monthly_data, raw_factor


def norm_df(hs300_trading_monthly, hs300_trade_data, hs300_wgt_data, raw_factor):
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


def get_adjust_weight(wgt_opt_df, trade_data, predate, curdate):
    df = wgt_opt_df[wgt_opt_df.trade_date == predate]
    code_list = df['ts_code'].drop_duplicates().to_list()
    trade_data = trade_data[trade_data.trade_date == curdate]
    close_price = get_stock_close_price(trade_data)
    close_price_filter = {}
    for code in code_list:
        close_price_filter[code] = close_price[code]
    position = get_position(df)
    adjust_weight = compute_weight_by_position(position, close_price_filter)
    return adjust_weight


def compute_position(close_price, weight):
    position = {}
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


def main():
    config_path = r'./config/config.json'
    config = Config(config_path)
    opt_monthly_data_path = config.opt_monthly_data_path
    trade_monthly_data_path = config.trade_monthly_data_path
    raw_factor_path = config.raw_factor_path
    daily_data_path = config.daily_data_path
    # fixed_wgt_data_path = config.fixed_weight_data_path
    fixed_wgt_data_path = config.fixed_opt_weight_data_path
    mrawret_save_path = config.mrawret_save_path
    image_save_path = config.image_save_path
    opt_weight_save_path = config.opt_weight_save_path
    opt2trade = config.opt2trade
    opt_monthly_data, trade_monthly_data, raw_factor = data_process(opt_monthly_data_path,
                                                                    raw_factor_path,
                                                                    trade_monthly_data_path)
    trade_date_list = trade_monthly_data['trade_date'].drop_duplicates().to_list()
    opt_date_list = opt_monthly_data['trade_date'].drop_duplicates().to_list()
    # opt2trade = load_obj(opt2trade)
    daily_data = read_csv(daily_data_path, sep='\t')
    wgt_data = pd.read_csv(fixed_wgt_data_path)
    wgt_opt_df = pd.DataFrame()
    mrawret = init_ans_df(trade_date_list)
    pre_trade_date = None
    adjust_weight = None
    backtest = BackTestHs300(config=config)
    for idx, opt_date in enumerate(opt_date_list[:-1]):
        if opt_date < 20050429:
            continue
        trade_date = opt_date
        specific_day_factor_data = raw_factor[raw_factor.trade_date == opt_date]
        specific_day_wgt_data = wgt_data[wgt_data.trade_date == opt_date]
        specific_day_daily_data = daily_data[daily_data.trade_date == trade_date]
        specific_day_trade_data = trade_monthly_data[trade_monthly_data.trade_date == trade_date]
        if pre_trade_date:
            adjust_weight = get_adjust_weight(wgt_opt_df, trade_monthly_data, pre_trade_date, trade_date)
        opt_weight, original_weight, vars_opt = backtest.optimize(
            specific_day_factor_data, specific_day_daily_data, specific_day_wgt_data, adjust_weight, trade_date)
        close_price = get_stock_close_price(specific_day_trade_data)
        stock_position = compute_position(close_price, opt_weight)
        for key, val in stock_position.items():
            vars_opt.loc[key, 'position'] = val
        if wgt_opt_df.empty:
            wgt_opt_df = vars_opt
        else:
            wgt_opt_df = pd.concat([wgt_opt_df, vars_opt], axis=0)

        opt_next_date = opt_date_list[idx+1]
        # trade_next_date = opt2trade[opt_next_date]
        trade_next_date = opt_next_date
        pre_trade_date = trade_date
        next_trade_day_data = trade_monthly_data[trade_monthly_data.trade_date == trade_next_date]
        next_trade_day_data = next_trade_day_data[['ts_code', '1mon_tradingreturn']]
        specific_day_wgt_data = specific_day_wgt_data.merge(next_trade_day_data, how='left', on=['ts_code'])
        mon_trade_return = backtest.get_factor_score(specific_day_wgt_data, '1mon_tradingreturn')
        mrawret.loc[idx+1, 'factor_model'] = dict_prod(opt_weight, mon_trade_return)
        mrawret.loc[idx+1, 'hs300index'] = dict_prod(original_weight, mon_trade_return)
        mrawret.loc[idx+1, 'net_ret'] = mrawret.loc[idx+1, 'factor_model'] - mrawret.loc[idx+1, 'hs300index']
        mrawret.loc[idx+1, 'factor_model_cum'] = mrawret.loc[idx, 'factor_model_cum'] * \
                                                (1 + mrawret.loc[idx+1, 'factor_model'])
        mrawret.loc[idx+1, 'hs300index_cum'] = mrawret.loc[idx, 'hs300index_cum'] * \
                                               (1 + mrawret.loc[idx+1, 'hs300index'])
        mrawret.loc[idx+1, 'net_ret_cum'] = mrawret.loc[idx, 'net_ret_cum'] * \
                                           (1 + mrawret.loc[idx+1, 'net_ret'])
    wgt_opt_df.to_csv(opt_weight_save_path, sep=',')
    print(sum(mrawret.loc[1:, 'net_ret'] >= 0)/len(mrawret.loc[1:, 'net_ret']))
    print(np.mean(mrawret.loc[1:, 'net_ret']) * 12)
    mrawret.to_csv(mrawret_save_path, sep=',',index=False)
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
    fig.savefig(image_save_path, dpi=100)


if __name__ == '__main__':
    main()