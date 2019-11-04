import gurobipy
import pandas as pd
import numpy as np
from tools import dict_prod


class BackTest(object):

    def __init__(self, config):
        pass

    def init_model(self, **kwargs):
        pass

    def add_constr(self, **kwargs):
        pass

    def record_opt_weight(self, **kwargs):
        pass

    def plot(self):
        pass

    @staticmethod
    def get_factor_score(factor_data, factor_name):
        df = factor_data[['ts_code', factor_name]]
        factor_score = {}
        for i in range(len(df)):
            tmp = df.iloc[i]
            code = tmp['ts_code']
            score = tmp[factor_name]
            factor_score[code] = score
        return factor_score

    @staticmethod
    def select_suspend_stock(all_stock, hs300_stock):
        all_stock = set(all_stock)
        hs300_stock = set(hs300_stock)
        suspend_stock = hs300_stock.difference(all_stock)
        return suspend_stock

    @staticmethod
    def get_ts_code_original_weight(hs300_wgt_data, date=None):
        if date:
            hs300_wgt_data = hs300_wgt_data[hs300_wgt_data.trade_date == date]
        original_weight = {}
        num = len(hs300_wgt_data)
        date_list = hs300_wgt_data['trade_date'].drop_duplicates().to_list()
        assert len(date_list) == 1, print('the hs300_wgt_data has more one days data'.format(date_list))
        assert num >= 280, print('the num of hs300 is not 300 in date {} '.format(date_list),
                                 'actual num is {}'.format(num))
        for i in range(num):
            tmp = hs300_wgt_data.iloc[i]
            ts_code = tmp['ts_code']
            weight = tmp['weight']
            assert ts_code is not None, print('check the ts_code in date {} code is {}'.format(date, ts_code))
            assert weight is not None, print('check the weight in date {} code is {}'.format(date, weight))
            original_weight[ts_code] = weight / 100
        assert abs(1 - sum(original_weight.values())) <= 0.01, print(sum(original_weight.values()))
        return original_weight


class BackTestHs300(BackTest):

    def __init__(self, config):
        super(BackTestHs300).__init__()
        self.name = 'hs300'
        self.constr_factor = config.constr_factor
        self.stock_lower_bound = config.stock_lower_bound
        self.stock_upper_bound = config.stock_upper_bound
        self.factor_upper_bound = config.factor_upper_bound
        self.factor_lower_bound = config.factor_lower_bound
        self.industry_lower_bound = config.industry_lower_bound
        self.industry_upper_bound = config.industry_upper_bound

    def init_model(self, hs300_ts_code, original_weight,
                   overall_score, adjust_weight, suspend_stock):
        stock_lower_bound = self.stock_lower_bound
        stock_upper_bound = self.stock_upper_bound
        model = gurobipy.Model(self.name)
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

    def init_ans_df(self, date_list):
        ans = pd.DataFrame({'trade_date': date_list})
        ans['factor_model'] = np.nan
        ans['hs300index'] = np.nan
        ans['net_ret'] = np.nan
        ans.loc[0, ['factor_model', 'hs300index', 'net_ret']] = 0

        ans['factor_model_cum'] = np.nan
        ans['hs300index_cum'] = np.nan
        ans['net_ret_cum'] = np.nan
        ans.loc[0, ['factor_model_cum', 'hs300index_cum', 'net_ret_cum']] = 1000
        return ans

    def add_trade_constr(self, model, weight, weight_binary,
                         suspend_stock, adjust_weight):
        if not adjust_weight:
            adjust_weight = {}
            for stock in suspend_stock:
                adjust_weight[stock] = 0
        for stock in suspend_stock:
            if stock in adjust_weight and adjust_weight[stock] != 0:
                model.addConstr(weight[stock] == adjust_weight[stock], name='trade_cons_' + stock)

    def add_stock_constr(self, model, weight, weight_binary, hs300_ts_code):
        # 选股数量约束，要求 [0，60]
        model.addConstrs(weight[j] <= weight_binary[j] for j in hs300_ts_code)
        model.addConstr(gurobipy.quicksum(weight_binary) <= 100, name='stock num')
        # 权重和约束，weight和为 1
        model.addConstr(gurobipy.quicksum(weight) == 1, 'budge')

    def add_industry_constr(self, model, weight, factor_data, suspend_stock_code, adjust_weight):
        industry_lower_bound = self.industry_lower_bound
        industry_upper_bound = self.industry_upper_bound
        all_industry = factor_data['gics_code'].drop_duplicates()
        all_industry = all_industry.to_list()
        all_industry.sort()
        for industry in all_industry:
            specific_industry_df = factor_data[factor_data.gics_code == 20.0]
            if len(specific_industry_df) <= 5:
                continue
            codes = specific_industry_df.ts_code.to_list()
            codes.sort()
            assert len(codes) == len(set(codes)), print('codes num and set codes num',
                                                        len(codes), len(set(codes)), industry)
            codes = set(codes)
            lb = sum(specific_industry_df.weight) / 100 - industry_lower_bound
            ub = sum(specific_industry_df.weight) / 100 + industry_upper_bound
            opt_industry_weight = sum([weight[j] for j in codes])
            model.addConstr(opt_industry_weight <= ub, name='industry_upper_bound: ' + str(industry))
            model.addConstr(opt_industry_weight >= lb, name='industry_lowwer_bound: ' + str(industry))

    def add_factor_constr(self, model, weight, factor_data, original_weight, factor_name):
        factor_upper_bound = self.factor_upper_bound
        factor_lower_bound = self.factor_lower_bound
        factor_score = self.get_factor_score(factor_data, factor_name)
        factor_weigth_score = dict_prod(original_weight, factor_score)
        upper_bound = factor_weigth_score * (1 + factor_upper_bound)
        lower_bound = factor_weigth_score * (1 + factor_lower_bound)
        model.addConstr(weight.prod(factor_score) >= lower_bound)
        model.addConstr(weight.prod(factor_score) <= upper_bound)

    def optimize(self, factor_data, specific_day_trade_data,
                 specific_day_hs300_wgt, adjust_weight, date):
        overall_score = self.get_factor_score(factor_data, 'overall_factor')
        trade_ts_codes = specific_day_trade_data['ts_code']
        hs300_ts_codes = specific_day_hs300_wgt['ts_code']
        suspend_stock_code = self.select_suspend_stock(trade_ts_codes, hs300_ts_codes)
        original_weight = self.get_ts_code_original_weight(specific_day_hs300_wgt)
        model, weight, weight_binary = self.init_model(hs300_ts_codes, original_weight,
                                                  overall_score, adjust_weight,
                                                  suspend_stock_code)
        self.add_stock_constr(model, weight, weight_binary, hs300_ts_codes)
        self.add_trade_constr(model, weight, weight_binary, suspend_stock_code, adjust_weight)
        self.add_industry_constr(model, weight, factor_data, suspend_stock_code, adjust_weight)
        for fac in self.constr_factor:
            self.add_factor_constr(model, weight, factor_data, original_weight, fac)
        model.optimize()
        vars_opt = pd.DataFrame(index=hs300_ts_codes.to_list())
        vars_opt['trade_date'] = date
        opt_weight = {}
        if model.status == gurobipy.GRB.Status.OPTIMAL:
            for v in model.getVars():
                varname = v.varname
                varname = varname.split('_')
                ts_code = varname[-1]
                colunm_name = varname[0]
                if colunm_name == 'optweight':
                    opt_weight[ts_code] = v.x
                vars_opt.loc[ts_code, colunm_name] = v.x
                vars_opt.loc[ts_code, 'ts_code'] = ts_code
        else:
            print('data {} infeasible'.format(date))
            model.computeIIS()
            model.write("./ilp/model_{}.ilp".format(date))
            exit()
        return opt_weight, vars_opt

