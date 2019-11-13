import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from configuration import Config
from statsmodels.distributions.empirical_distribution import ECDF


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


def compute(x, nagtive=False):
    if len(x) == 0:
        return x
    sd_szie = stand(x, 0.05)
    if nagtive:
        sd_szie = - sd_szie
    sd_sd_size = std_winsor(sd_szie)
    try:
        size_cdf = ECDF(sd_sd_size)
    except:
        print(x)
        exit()
    size_cdf_ = size_cdf(sd_sd_size)
    return size_cdf_


def data_process_v0():
    data_path = r'./data/Data_industry.csv'
    data_df = pd.read_csv(data_path)
    data_df = data_df[data_df.Date>=20100226]
    data_df = data_df.drop_duplicates()
    factor_date = data_df['Date'].drop_duplicates()
    factor_all_industry = data_df['FirstIndustryCode'].drop_duplicates()
    return data_df, factor_date, factor_all_industry


def compute_factor_v0(data_df,date_factor,industry_factor):
    raw_factor = None
    cnt = 0
    for i in industry_factor:
        print('\n', 'current industry is : ', i, '\n')
        for j in tqdm(date_factor):
            raw_file1 = data_df[data_df.Date == j]
            raw_file1 = raw_file1[raw_file1.FirstIndustryCode == i]
            # print(raw_file1.columns)
            if raw_file1.empty:
                cnt += 1
                continue
            else:
                # size
                raw_file1["size_factor"] = compute(raw_file1['Log_mkt_Cap'], nagtive=True)
                # Volatility（波动率）
                raw_file1["vol_factor"] = compute(raw_file1['Volatility60'], nagtive=True)
                # Momentum
                raw_file1["mo_factor"] = compute(raw_file1['12Mo_1Mo'])
                # Quality
                sd_roa = stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 资产回报
                roa_cdf = ECDF(sd_roa)
                sd_acc = stand(raw_file1["Accural"], 0.05)  # accural ？现金流
                acc_cdf = ECDF(sd_acc)
                cash_debt = raw_file1["NetOperateCashFlow"] / raw_file1["TotalLiability"]  # 总负债
                sd_cash = stand(cash_debt, 0.05)
                cash_cdf = ECDF(sd_cash)
                raw_file1["qa_factor"] = 0.25 * roa_cdf(sd_roa) + 0.25 * acc_cdf(sd_acc) + 0.5 * cash_cdf(sd_cash)
                # Value
                sd_cashval = stand(raw_file1['Cash_Over_MktCap'], 0.05)  # 现金除以市值
                sd_sd_cashval = std_winsor(sd_cashval)
                cashval_cdf = ECDF(sd_sd_cashval)
                sd_roa = stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 收益除以市值
                sd_sd_roa = std_winsor(sd_roa)
                roa_cdf = ECDF(sd_sd_roa)
                sd_bp = stand(raw_file1['BP2'], 0.05)  # 市净率
                sd_sd_bp = std_winsor(sd_bp)
                bp_cdf = ECDF(sd_sd_bp)
                raw_file1["value_factor"] = 1 / 3 * cashval_cdf(sd_sd_cashval) \
                                            + 1 / 3 * roa_cdf(sd_sd_roa) \
                                            + 1 / 3 * bp_cdf(sd_sd_bp)
                raw_file1["overall_factor"] = raw_file1["value_factor"] \
                                              + raw_file1["qa_factor"] \
                                              + raw_file1["mo_factor"] \
                                              + raw_file1["vol_factor"] \
                                              + raw_file1["size_factor"] * 0.15
                if not raw_factor:
                    raw_factor = raw_file1
                else:
                    raw_factor = pd.concat([raw_factor, raw_file1], axis=0)
    print("wrong count of data is ", cnt)
    raw_factor.to_csv('raw_factor_test.csv', mode='w', header=True)


def data_process(config):
    all_data_path = config.all_data_path
    opt_trading_monthly_path = config.opt_monthly_data_path
    all_data = pd.read_csv(all_data_path, sep=',')
    opt_trading_monthly = pd.read_csv(opt_trading_monthly_path, sep=r',')
    hs300_momentum = opt_trading_monthly[['trade_date', 'ts_code', 'last_1mon_pricechange', '12monPC_1monPC']]
    all_data = all_data.merge(hs300_momentum, how='left', on=["trade_date", 'ts_code'])
    date_list = all_data['trade_date'].drop_duplicates()
    industry_list = all_data['gics_code'].drop_duplicates()
    return all_data, date_list, industry_list


def compute_factor(data_df, date_list, industry_list, config):
    raw_factor = pd.DataFrame()
    cnt = 0
    raw_factor_path = config.raw_factor_path
    for i in industry_list:
        print('current industry is : ', i)
        for j in tqdm(date_list):
            raw_file1 = data_df[data_df.trade_date == j]
            raw_file1 = raw_file1[raw_file1.gics_code == i]
            # print(raw_file1.columns)
            if raw_file1.empty:
                cnt += 1
                continue
            else:
                # size
                raw_file1["size_factor"] = compute(raw_file1['Log_mkt_Cap'], nagtive=True)
                # Volatility（波动率）
                raw_file1["volatility_factor"] = compute(raw_file1['Volatility'], nagtive=True)
                # Idiosyncratic volatility
                # raw_file1['idioVolatility_Factor'] = compute(raw_file1['idio_vol'], nagtive=True)
                # RSI 过去n：14天内多少天下降，多少天上升
                raw_file1['RSI_factor'] = compute(raw_file1['RSI'], nagtive=True)
                # Momentum
                raw_file1["momentum_factor"] = compute(raw_file1['last_1mon_pricechange'], nagtive=True)
                # Quality
                sd_roa = stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 资产回报
                roa_cdf = ECDF(sd_roa)
                sd_acc = stand(raw_file1["q_opincome"], 0.05)  # accural ？现金流
                acc_cdf = ECDF(sd_acc)
                sd_nocfod = stand(raw_file1['NOCF_Over_Debt'], 0.05)
                nocfod_cdf = ECDF(sd_nocfod)
                raw_file1["quality_factor"] = 0.25 * roa_cdf(sd_roa) + 0.25 * acc_cdf(sd_acc) + 0.5 * nocfod_cdf(sd_nocfod)
                # Value
                sd_cashval = stand(raw_file1['Cash_Over_MktCap'], 0.05)  # 现金除以市值
                sd_sd_cashval = std_winsor(sd_cashval)
                cashval_cdf = ECDF(sd_sd_cashval)
                sd_roa = stand(raw_file1['Rev_Over_mktCap'], 0.05)  # 收益除以市值
                sd_sd_roa = std_winsor(sd_roa)
                roa_cdf = ECDF(sd_sd_roa)
                sd_bp = stand(raw_file1['pb'], 0.05)  # 市净率
                sd_sd_bp = std_winsor(sd_bp)
                bp_cdf = ECDF(sd_sd_bp)

                raw_file1["value_factor"] = 1 / 3 * cashval_cdf(sd_sd_cashval) \
                                            + 1 / 3 * roa_cdf(sd_sd_roa) \
                                            + 1 / 3 * bp_cdf(sd_sd_bp)

                raw_file1["overall_factor"] = raw_file1["size_factor"] * 0.15 \
                                              + raw_file1["volatility_factor"] * 0.5 \
                                              + raw_file1["RSI_factor"] * 0.5 \
                                              + raw_file1["momentum_factor"] * 0.5 \
                                              + raw_file1["quality_factor"] \
                                              + raw_file1["value_factor"]
                if raw_factor.empty:
                    raw_factor = raw_file1
                else:
                    raw_factor = pd.concat([raw_factor, raw_file1], axis=0)
    print("wrong count of data is ", cnt)
    raw_factor.to_csv(raw_factor_path, mode='w', header=True)


def main():
    start = time.time()
    config_data_path = r'./config/config.json'
    config = Config(config_data_path)
    data_df, date_list, industry_list = data_process(config)
    date_factor = date_list.sort_values()
    industry_factor = industry_list.sort_values()
    compute_factor(data_df, date_factor, industry_factor, config)
    print('cost time:', time.time()-start)


if __name__ == '__main__':
    main()
