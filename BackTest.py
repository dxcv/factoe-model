import gurobipy
import pandas as pd
from collections import Counter
import numpy as np
from tools import dict_prod
from data_process import load_obj


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
    def get_suspend_stock(all_stock, hs300_stock):
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

    @staticmethod
    def check_code_list(opt_code_list, trade_code_list):
        assert set(opt_code_list) == set(trade_code_list), print(set(opt_code_list).difference(set(trade_code_list)))


class BackTestHs300(BackTest):

    def __init__(self, config):
        super(BackTestHs300).__init__()
        self.name = 'hs300'
        self.constr_factor = config.constr_factor
        self.stock_number = config.stock_number
        self.stock_lower_bound = config.stock_lower_bound
        self.stock_upper_bound = config.stock_upper_bound
        self.factor_upper_bound = config.factor_upper_bound
        self.factor_lower_bound = config.factor_lower_bound
        self.industry_lower_bound = config.industry_lower_bound
        self.industry_upper_bound = config.industry_upper_bound
        self.name2code = load_obj(config.name2code)
        self.code2name = dict((v,k) for k,v in self.name2code.items())
        self.gics2name = load_obj(config.gics2name)

    def init_model(self, wgt_code_list, original_weight,
                   overall_score, adjust_weight, suspend_stock):
        stock_lower_bound = self.stock_lower_bound
        stock_upper_bound = self.stock_upper_bound
        model = gurobipy.Model(self.name)
        weight, weight_binary = {}, {}
        for code in wgt_code_list:
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

    def add_trade_constr(self, model, weight, weight_binary,
                         suspend_stock, adjust_weight):
        if not adjust_weight:
            return
        for stock in suspend_stock:
            if stock in adjust_weight and adjust_weight[stock] != 0:
                model.addConstr(weight[stock] == adjust_weight[stock], name='trade_cons_' + stock)

    def add_stock_number_constr(self, model, weight, weight_binary, code_list):
        stock_number = self.stock_number
        model.addConstrs(weight[j] <= weight_binary[j] for j in code_list)
        model.addConstr(gurobipy.quicksum(weight_binary) <= stock_number, name='stock num')
        model.addConstr(gurobipy.quicksum(weight) == 1, 'budge')

    def add_industry_constr(self, model, weight, daily_data, suspend_stock_code, adjust_weight):
        industry_lower_bound = self.industry_lower_bound
        industry_upper_bound = self.industry_upper_bound
        all_industry = daily_data['gics_code'].drop_duplicates().to_list()
        all_industry.sort()
        for industry in all_industry:
            specific_industry_df = daily_data[daily_data.gics_code == industry]
            codes = specific_industry_df.ts_code.to_list()
            suspend_stock_in_specific_industry = set(codes).union(suspend_stock_code)
            if adjust_weight:
                suspend_stock_weight_sum = sum([adjust_weight[j] for j in suspend_stock_in_specific_industry if j in adjust_weight])
            else:
                suspend_stock_weight_sum = 0
            if len(specific_industry_df) <= 5:
                continue
            assert len(codes) == len(set(codes)), \
                print('codes num and set codes num', len(codes), len(set(codes)), industry)
            lb = sum(specific_industry_df.weight) / 100 - industry_lower_bound
            ub = sum(specific_industry_df.weight) / 100 + industry_upper_bound
            ub_adjust = suspend_stock_weight_sum + industry_upper_bound
            if ub <= suspend_stock_weight_sum:
                ub = ub_adjust
            opt_industry_weight = sum([weight[j] for j in codes])
            model.addConstr(opt_industry_weight <= ub, name='industry_upper_bound: ' + str(self.gics2name[industry]))
            model.addConstr(opt_industry_weight >= lb, name='industry_lowwer_bound: ' + str(self.gics2name[industry]))

    def add_factor_constr(self, model, weight, factor_data, original_weight, factor_name):
        factor_upper_bound = self.factor_upper_bound
        factor_lower_bound = self.factor_lower_bound
        factor_score = self.get_factor_score(factor_data, factor_name)
        factor_weigth_score = dict_prod(original_weight, factor_score)
        upper_bound = factor_weigth_score * (1 + factor_upper_bound)
        lower_bound = factor_weigth_score * (1 + factor_lower_bound)
        model.addConstr(weight.prod(factor_score) >= lower_bound)
        model.addConstr(weight.prod(factor_score) <= upper_bound)

    def optimize(self, specific_day_factor_data, specific_day_daily_data,
                 specific_day_wgt_data, adjust_weight, trade_date):
        trade_day_data = specific_day_daily_data
        opt_day_data = specific_day_factor_data
        overall_score = self.get_factor_score(opt_day_data, 'overall_factor')

        trade_code_list = trade_day_data['ts_code'].to_list()
        wgt_code_list = specific_day_wgt_data['ts_code'].to_list()
        opt_code_list = opt_day_data['ts_code'].to_list()
        suspend_stock_code = self.get_suspend_stock(trade_code_list, wgt_code_list)
        original_weight = self.get_ts_code_original_weight(specific_day_wgt_data)
        model, weight, weight_binary = self.init_model(wgt_code_list, original_weight,
                                                       overall_score, adjust_weight,
                                                       suspend_stock_code)
        self.add_stock_number_constr(model, weight, weight_binary, wgt_code_list)
        self.add_trade_constr(model, weight, weight_binary, suspend_stock_code, adjust_weight)
        self.add_industry_constr(model, weight, specific_day_wgt_data, suspend_stock_code, adjust_weight)
        for fac in self.constr_factor:
            self.add_factor_constr(model, weight, opt_day_data, original_weight, fac)
        model.optimize()
        vars_opt = pd.DataFrame(index=wgt_code_list)
        vars_opt['trade_date'] = trade_date
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
            print('trade date {} infeasible'.format(trade_date))
            model.computeIIS()
            model.write("./ilp/model_{}.ilp".format(trade_date))
            exit()
        return opt_weight, original_weight, vars_opt

