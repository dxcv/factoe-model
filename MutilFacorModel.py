from data_process import *
from toy_backtest_v2 import backtest
from Factor import *


class FactorModel(object):
    def __init__(self, config):
        pass

    def norm_data(self, **kwargs):
        pass

    def data_process(self):
        pass

    def factor_compute(self):
        pass

    def backtest(self):
        pass

    def process_momenta(self):
        pass


class FactorModelHs300(FactorModel):

    def __init__(self, config):
        super(FactorModel, self).__init__()
        self.config = config
        self.index_name = config.index_name
        self.tmp_file_path = config.tmp_path
        self.raw_factor = pd.DataFrame()
        self.monthly_data_path = None
        self.wgt_data_path = None
        self.trading_data = None
        self.weight_data = None
        self.date_list = None
        self.code2name = None
        self.code2gics = None
        self.gics2name = None

    def norm_data(self, trading_data, weight_data, date_list):
        self.trading_data = trading_data
        self.weight_data = weight_data
        self.date_list = date_list

    def first_stage_prepare(self):
        self.code2name = get_name2code(self.weight_data)
        self.code2gics = get_code2gics()
        self.gics2name = get_gics2name()

    def read_finacial_data(self):
        config = self.config
        balance_data_path = config.balance_data_path
        income_data_path = config.income_data_path
        cashflow_data_path = config.cashflow_data_path
        fina_indicator_data_path = config.fina_indicator_data_path
        balancesheetdata = read_csv(balance_data_path, sep='\t')
        incomedata = read_csv(income_data_path, sep='\t')
        cashflowdata = read_csv(cashflow_data_path, sep='\t')
        fina_indicatordata = read_csv(fina_indicator_data_path, sep='\t')
        return balancesheetdata,incomedata,cashflowdata,fina_indicatordata

    def prepare_finacial_data(
            self, balancesheetdata, incomedata, cashflowdata,fina_indicatordata):
        trading_data = self.trading_data
        config = self.config
        vol_lag = config.vol_lag
        rsi_lag = config.rsi_lag
        wgt_df = prepare_hs300_wgt_data_df(self.weight_data)
        vol_data = prepare_vol_data_df(trading_data, vol_lag)
        rsi_data = prepare_rsi_data_df(trading_data, rsi_lag)
        all_sheet = prepare_full_final_sheet(balancesheetdata, incomedata, cashflowdata, trading_data)
        finacial_indicator_data = prepare_indicator_sheet(fina_indicatordata, trading_data, 'indicator')
        return wgt_df, vol_data, rsi_data, all_sheet, finacial_indicator_data

    def data_process(self):
        config = self.config
        trading_data = self.trading_data
        balancesheetdata, incomedata, cashflowdata, fina_indicatordata = self.read_finacial_data()
        month_const = config.month_const
        lag_num = config.lag_num
        wgt_df, vol_data, rsi_data, all_sheet, finacial_indicator_data = self.prepare_finacial_data(
            balancesheetdata, incomedata, cashflowdata, fina_indicatordata
        )
        other_data = trading_data[['ts_code', 'trade_date', 'total_mv', 'pb', 'close']]
        wgt_data = wgt_df.merge(vol_data, how='left', on=['ts_code', 'trade_date'])
        wgt_data = wgt_data.merge(rsi_data, how='left', on=['ts_code', 'trade_date'])
        wgt_data = wgt_data.merge(all_sheet, how='left', on=['ts_code', 'trade_date'])
        wgt_data = wgt_data.merge(finacial_indicator_data, how='left', on=['ts_code', 'trade_date'])
        wgt_data = wgt_data.merge(other_data, how='left', on=['ts_code', 'trade_date'])
        wgt_data["Log_mkt_Cap"] = np.log(list(wgt_data["total_mv"]))
        wgt_data["Cash_Over_MktCap"] = wgt_data.fcff.div(wgt_data.total_mv, axis=0)
        wgt_data["Rev_Over_mktCap"] = wgt_data.revenue.div(wgt_data.total_mv, axis=0)
        wgt_data["NOCF_Over_Debt"] = wgt_data.n_cashflow_act.div(wgt_data.total_liab, axis=0)
        self.wgt_data_path = os.path.join(self.tmp_file_path, config.wgt_trading_data_monthly)
        wgt_data.to_csv(self.wgt_data_path, index=False)

        all_monthly_data = prepare_all_date_monthly_data_df()
        all_monthly_data = norm_df(all_monthly_data)
        all_monthly_trading_date = all_monthly_data.merge(trading_data, how='left', on=['ts_code', 'trade_date'])
        monthly_chg_data = prepare_monthly_price_change_data(all_monthly_trading_date, month_const, lag_num)
        all_monthly_trading_date = all_monthly_trading_date.merge(
            monthly_chg_data, how='left',on=['ts_code', 'trade_date'])
        all_monthly_trading_date["12monPC_1monPC"] = all_monthly_trading_date["last_12mon_pricechange_lag"] - \
                                                     all_monthly_trading_date["last_1mon_pricechange"]
        self.monthly_data_path = os.path.join(self.tmp_file_path, 'monthly_data.csv')
        all_monthly_trading_date.to_csv(self.monthly_data_path, index=False)

    def factor_compute(self):
        monthly_data = pd.read_csv(self.monthly_data_path)
        wgt_data = pd.read_csv(self.wgt_data_path)
        hs300_momentum = monthly_data[['trade_date', 'ts_code', 'last_1mon_pricechange', '12monPC_1monPC']]
        all_data = wgt_data.merge(hs300_momentum, how='left', on=["trade_date", 'ts_code'])
        date_list = all_data['trade_date'].drop_duplicates()
        industry_list = all_data['gics_code'].drop_duplicates()
        size_factor = SizeFactor('size_factor')
        volatility_factor = VolatilityFactor('volatility_factor')
        RSI_factor = RsiFactor('RSI_factor')
        momentum_factor = MomentaFactor('momentum_factor')
        quality_factor = QualityFactor('quality_factor')
        value_factor = ValueFactor('value_factor')
        for i in industry_list:
            print('current industry is : ', self.gics2name[i])
            for j in date_list:
                raw_data = all_data[wgt_data.trade_date == j]
                raw_data = raw_data[raw_data.gics_code == i]
                raw_data['size_factor'] = size_factor.data_procsess(raw_data['Log_mkt_Cap'])
                raw_data['volatility_factor'] = volatility_factor.data_procsess(raw_data['Volatility'])
                raw_data['RSI_factor'] = RSI_factor.data_procsess(raw_data['RSI'], nagtive=True)
                raw_data["momentum_factor"] = momentum_factor.data_procsess(raw_data['last_1mon_pricechange'],
                                                                            nagtive=True)
                quality_factor.data_procsess(raw_data)
                value_factor.data_procsess(raw_data)
                raw_data["overall_factor"] = raw_data["size_factor"] * 0.15 \
                                              + raw_data["volatility_factor"] * 0.5 \
                                              + raw_data["RSI_factor"] * 0.5 \
                                              + raw_data["momentum_factor"] * 0.5 \
                                              + raw_data["quality_factor"] \
                                              + raw_data["value_factor"]
                if self.raw_factor.empty:
                    self.raw_factor = raw_data
                else:
                    self.raw_factor = pd.concat([self.raw_factor, raw_data], axis=0)

    def backtest(self):
        backtest()


