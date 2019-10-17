#!/usr/bin/python3

"""
Date: 2019-09-03
Author: Teng
Notation: This program is used to construct the framework of the factor model and calculate the back-testing results
for the relative model
"""

# import the necessary packages
import pandas as pd
import numpy as np
import math
import copy

from gurobipy import *
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime

# set the constant variables
w_ub = 0.04                              # the upper bound from HS300 index for assets' weight
w_lb = 0.04                              # the lower bound from HS300 index for assets' weight

M_ub = 60                                # the max number of assets in the solution
m_lb = 1                                 # the min number of assets in the solution

fmin = 0.001                             # default: 0.1%  set the minimal nonzero fraction of assets
fmax = 1.0                               # default: 1.0   set the maximal nonzero fraction of assets

w_ind_ub = 0.02                          # the upper bound from HS300 index for sum of assets' weight at the industry
w_ind_lb = 0.02                          # the lower bound from HS300 index for sum of assets' weight at the industry

w_fac_ub = 0.35                          # the upper bound from HS300 index for weighted sum of the assets' score
w_fac_lb = 0.07                          # the lower bound from HS300 index for weighted sum of the assets' score

# Step One: input the initial necessary data
# 1.1 the monthly trading data of stocks in HS300 from 20020101 to 20190806
HS300_tradingdata = pd.read_csv(r'./data/HS300tradingmonthly.txt',
                                  header='infer', encoding=None,  sep=',')

# 1.2 input the factor data
RawFactor = pd.read_csv(r'./data/raw_factor2_5.csv',
                          header='infer', encoding=None, sep=',')

# Step Two: calculate the return of the factor model and HS300 index strategy
# 2.1 define the data frame for the returns for different models
mDate = list(set(RawFactor.Date))
mDate.sort()

mrawret = pd.DataFrame({'Date': mDate})
mrawret['FactorModel'] = np.nan
mrawret['HS300Index'] = np.nan
mrawret['NetRet'] = np.nan
mrawret.loc[0, ['FactorModel', 'HS300Index', 'NetRet']] = 0

mrawret['FactorModel_cum'] = np.nan
mrawret['HS300Index_cum'] = np.nan
mrawret['NetRet_cum'] = np.nan
mrawret.loc[0, ['FactorModel_cum', 'HS300Index_cum', 'NetRet_cum']] = 1000

# 2.2 calculate the returns for different models
for i in range(mrawret.shape[0] - 1):
    rawfactor_i = copy.deepcopy(RawFactor.loc[RawFactor.Date == mrawret.loc[i, 'Date']].reset_index(drop=True))
    N = rawfactor_i.shape[0]

    # Create an empty model
    m = Model('Factor Model')

    # Add a variable of weight and dummy for each stock
    vars = []
    for j in range(2*N):
        if j < N:
            lb_weight = max(0, rawfactor_i.loc[j, 'weight'] / 100 - w_lb)
            ub_weight = min(1, rawfactor_i.loc[j, 'weight'] / 100 + w_ub)
            vars.append(m.addVar(lb=lb_weight, ub=ub_weight, vtype=GRB.CONTINUOUS))
        else:
            vars.append(m.addVar(lb=0, ub=1, vtype=GRB.INTEGER))

    vars = pd.Series(vars)

    # Set objective
    obj_factor = list(rawfactor_i.Size_Factor + rawfactor_i.Quality_Factor)
    obj_vector = pd.Series((obj_factor+[0]*N))
    m.setObjective(obj_vector.dot(vars), GRB.MAXIMIZE)

    # set constraint for the fixed budget
    m.addConstr(sum(vars.loc[range(N)]) == 1, 'budget')

    # set constraint for the number of assets in the solution to be between m and M
    m.addConstr(sum(vars.loc[range(N, 2*N)]) <= M_ub, 'UpperNumber')
    m.addConstr(sum(vars.loc[range(N, 2*N)]) >= m_lb, 'lowerNumber')

    # set the semicontinuous constraints
    for k in range(N):
        m.addConstr(vars.loc[k] <= fmax * vars.loc[k + N], 'UpperBound_dis : ' + str(k))
        m.addConstr(vars.loc[k] >= fmin * vars.loc[k + N], 'LowerBound_dis : ' + str(k))

    # set the constrain for the sum of weight for each industry
    all_industry = list(set(rawfactor_i.Gics))
    all_industry.sort()

    for indst in range(len(all_industry)):
        rawfactor_i_indst = copy.deepcopy(rawfactor_i.loc[rawfactor_i.Gics == all_industry[indst]])
        print(all_industry[indst])
        print(mrawret.loc[i, 'Date'])
        print(rawfactor_i_indst.index)
        exit()
        m.addConstr(sum(vars[rawfactor_i_indst.index]) <= sum(rawfactor_i_indst.weight)/100 + w_ind_ub,
                    'UpperBound_indst : ' + str(all_industry[indst]))

        m.addConstr(sum(vars[rawfactor_i_indst.index]) >= sum(rawfactor_i_indst.weight)/100 - w_ind_lb,
                    'LowerBound_indst : ' + str(all_industry[indst]))

    # set the constrain for the sum of score for each factor
    m.addConstr(rawfactor_i.Size_Factor.dot(vars.loc[range(N)]) <= rawfactor_i.Size_Factor.dot(rawfactor_i.weight)/100
                + w_fac_ub, 'size factor UpperBound')
    m.addConstr(rawfactor_i.Size_Factor.dot(vars.loc[range(N)]) >= rawfactor_i.Size_Factor.dot(rawfactor_i.weight)/100
                + w_fac_lb, 'size factor LowerBound')

    m.addConstr(rawfactor_i.IdioVolatility_Factor.dot(vars.loc[range(N)]) <= rawfactor_i.IdioVolatility_Factor.dot(rawfactor_i.weight)/100
                + w_fac_ub, 'IdioVolatility Factor UpperBound')
    m.addConstr(rawfactor_i.IdioVolatility_Factor.dot(vars.loc[range(N)]) >= rawfactor_i.IdioVolatility_Factor.dot(rawfactor_i.weight)/100
                + w_fac_lb, 'IdioVolatility Factor LowerBound')

    m.addConstr(rawfactor_i.RSI_Factor.dot(vars.loc[range(N)]) <= rawfactor_i.RSI_Factor.dot(rawfactor_i.weight)/100
                + w_fac_ub, 'RSI Factor UpperBound')
    m.addConstr(rawfactor_i.RSI_Factor.dot(vars.loc[range(N)]) >= rawfactor_i.RSI_Factor.dot(rawfactor_i.weight)/100
                + w_fac_lb, 'RSI_Factor LowerBound')

    m.addConstr(rawfactor_i.Quality_Factor.dot(vars.loc[range(N)]) <= rawfactor_i.Quality_Factor.dot(rawfactor_i.weight)/100
                + w_fac_ub, 'QualityFactor UpperBound')
    m.addConstr(rawfactor_i.Quality_Factor.dot(vars.loc[range(N)]) >= rawfactor_i.Quality_Factor.dot(rawfactor_i.weight)/100
                + w_fac_lb, 'Quality Factor LowerBound')

    m.addConstr(rawfactor_i.Value_Factor.dot(vars.loc[range(N)]) <= rawfactor_i.Value_Factor.dot(rawfactor_i.weight)/100
                + w_fac_ub, 'Value Factor UpperBound')
    m.addConstr(rawfactor_i.Value_Factor.dot(vars.loc[range(N)]) >= rawfactor_i.Value_Factor.dot(rawfactor_i.weight)/100
                + w_fac_lb, 'Value Factor LowerBound')

    # Optimize model to find the optimal stocks and the optimal weight
    m.optimize()

    # get the optimal weight and the optimal stocks
    vars_opt = pd.DataFrame()
    mcount = 0

    for v in m.getVars():
        vars_opt.loc[mcount, 'varname'] = v.varname
        vars_opt.loc[mcount, 'value'] = v.x
        mcount += 1
    vars_opt_weight = copy.deepcopy(vars_opt.loc[range(N), 'value'].reset_index(drop=True))
    vars_opt_dummy = copy.deepcopy(vars_opt.loc[range(N, 2*N), 'value'].reset_index(drop=True))

    #  get the return of next month and calculate the raw return
    rawfactor_i_next = copy.deepcopy(HS300_tradingdata.loc[HS300_tradingdata.Date == mrawret.loc[i+1, 'Date']].reset_index(drop=True))
    rawfactor_i = rawfactor_i.merge(rawfactor_i_next, how='left', left_on=["Codes"], right_on=["Codes"])
    mrawret.loc[i+1, 'FactorModel'] = vars_opt_weight.dot(rawfactor_i['1mon_tradingreturn'])
    mrawret.loc[i+1, 'HS300Index'] = rawfactor_i.weight.dot(rawfactor_i['1mon_tradingreturn'])/100
    mrawret.loc[i+1, 'NetRet'] = mrawret.loc[i+1, 'FactorModel'] - mrawret.loc[i+1, 'HS300Index']

    mrawret.loc[i+1, 'FactorModel_cum'] = mrawret.loc[i, 'FactorModel_cum'] * (1 + mrawret.loc[i+1, 'FactorModel'])
    mrawret.loc[i+1, 'HS300Index_cum'] = mrawret.loc[i, 'HS300Index_cum'] * (1 + mrawret.loc[i+1, 'HS300Index'])
    mrawret.loc[i+1, 'NetRet_cum'] = mrawret.loc[i, 'NetRet_cum'] * (1 + mrawret.loc[i+1, 'NetRet'])

    print(i)

# Step Three: plot the return of the factor model and the HS300 strategy
# 3.1 calculate the winning rate of the Factor Model and the return annually
print(sum(mrawret.loc[1:, 'NetRet'] >= 0)/len(mrawret.loc[1:, 'NetRet']))
print(np.mean(mrawret.loc[1:, 'NetRet']) * 12)

# 3.2 plot the cumulative return of factor model, HS300 index strategy and the net cumulative return
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

fig.savefig("./Factor_Model_Payoff_plot_Mateng.pdf", dpi=100)


