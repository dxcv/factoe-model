from configuration import Config
import pandas as pd
import numpy as np


def main():
    config_path = r'./config/config.json'
    config = Config(config_path)
    mrawret_save_path = config.mrawret_save_path
    mrawret = pd.read_csv(mrawret_save_path)
    num = len(mrawret)
    func = lambda x: (pow(x, 1/num) - 1) * 52
    factor_model_return = mrawret.loc[num-1, 'factor_model_cum'] / mrawret.loc[0, 'factor_model_cum']
    hs300index_return = mrawret.loc[num-1, 'hs300index_cum'] / mrawret.loc[0, 'hs300index_cum']
    net_ret_return = mrawret.loc[num-1, 'net_ret_cum'] / mrawret.loc[0, 'net_ret_cum']
    print('factor model ', func(factor_model_return))
    print('hs300index', func(hs300index_return))
    print("net_ret ", func(net_ret_return))
    print("*" * 50)
    print('factor model ', np.mean(mrawret.loc[1:, 'factor_model']) * 52)
    print('hs300index ', np.mean(mrawret.loc[1:, 'hs300index']) * 52)
    print('net_ret ', np.mean(mrawret.loc[1:, 'net_ret']) * 52)


if __name__ == '__main__':
    main()
