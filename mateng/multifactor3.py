#!/usr/bin/python3

"""
Created on Wed Aug 21 17:33:52 2019

@author: mateng

Notation: this program is a multifactor model based numerious trading and 

financial data of stocks in A share from Tushare and Wind

"""

# import the necessary packages
import pandas as pd
import numpy as np
import math
import copy
import statsmodels.api as sm

# set the constant variables
trim_perct = 5           # the cutoff percent for the percent winsorizing

std_num = 3              # the cutoff number of the standard deviation for winsorzing

listed_num = 1           # only choose the industry has at least 5 listed firms 

factor_names = ["Size_Factor", 'Volatility_Factor', 'IdioVolatility_Factor', 'RSI_Factor', 'Momentum_Factor', 
                'Quality_Factor', 'Value_Factor']  # the names of the five factors

Overall_weight = [0.15, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]        # the weight of five factors in the overall factor: factor_names

Quality_weight = [0.25, 0.25, 0.5]                 # the weight for the quality factor: Rev_Over_mktCap, Accural, NOCF_Over_Debt

Value_weight = [1/3, 1/3, 1/3]                     # the weight for the value factor: Cash_Over_MktCap, Rev_Over_mktCap, BP

vol_lag = 60             # the lag for the volatility

rsi_lag = 10             # the lag for the RSI

'''
#######################################################
############ Step One: input the initial necessary data

##1.1 the industry data of stocks in HS300 at 2019-08-08
HS300industry = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\HS300_industry.txt', 
                              header='infer', encoding='gbk', sep=',')

##1.2 the industry data of stocks in HS300 from 2002 to 2019 monthly
HS300industry_ts = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\HS300_industry_ts.txt', 
                              header='infer', encoding='gbk', sep=',')

##1.3 the weight data of stocks in HS300 from 20020101 to 20190806
HS300weight  = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_weight.txt', 
                              header='infer', encoding=None, delim_whitespace=True)

##1.4 the trading data of stocks in HS300 from 20020101 to 20190806
HS300_tradingdata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_tradingdata.txt', 
                              header='infer', encoding='gbk', delim_whitespace=True)

##1.5 the financial data of stocks in HS300 from 20020101 to 20190806
HS300_balancesheetdata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_balancesheetdata.txt', 
                                       header='infer', encoding='gbk', delim_whitespace=True)

HS300_incomedata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_incomedata.txt', 
                                 header='infer', encoding='gbk', delim_whitespace=True)

HS300_cashflowdata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_cashflowdata.txt', 
                                   header='infer', encoding='gbk', delim_whitespace=True)

HS300_fina_indicatordata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\399300.SZ_fina_indicatordata.txt', 
                                         header='infer', encoding='gbk', delim_whitespace=True)


###################################################################################
############ Step Two: basic processing for the above data od stocks in HS300 index

## 2.1 fill in the nan values in the time series data of industry
for i in range(HS300industry_ts.shape[0]):
    if np.isnan(HS300industry_ts.loc[i,'GicsCodes']):
        HS300industry_ts.loc[i,'Gics'] = list(HS300industry.loc[HS300industry.Codes == HS300industry_ts.loc[i,'Codes'], 'industry_gics'])
        HS300industry_ts.loc[i,'GicsCodes'] = list(HS300industry.loc[HS300industry.Codes == HS300industry_ts.loc[i,'Codes'], 'industry_gicscode'])
        print(str(i) + ' : ' + str(HS300industry_ts.loc[i,'Date'])+' '+ (HS300industry_ts.loc[i,'Codes'])  + 'has been delisted')


## 2.2 merge the time series data of industry with the weight data of index
HS300alldata = HS300industry_ts.merge(HS300weight, how='inner', left_on=['Date', 'Codes'], right_on=["trade_date", 'con_code']) 
HS300alldata = HS300alldata.drop(["trade_date", 'con_code'], axis = 1)

## 2.3 merge the industry data and the trading data of stocks in HS300 index
HS300alldata = HS300alldata.merge(HS300_tradingdata, how='left',
                                    left_on=["Date", 'Codes'], right_on=['trade_date', 'ts_code']) 


## 2.4 fill the nan values in HS300alldata with the last values
for i in range(HS300alldata.shape[0]):
    ## fill in the nan values with the last values
    if np.isnan(HS300alldata.loc[i, 'vol']):
        mtradingdata = copy.deepcopy(HS300_tradingdata.loc[HS300_tradingdata.ts_code == HS300alldata.loc[i, 'Codes']])
        mtradingdata_last = copy.deepcopy(mtradingdata.loc[mtradingdata.trade_date <= HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
        if mtradingdata_last.shape[0] > 0:
            HS300alldata.loc[i, list(mtradingdata_last.columns)] = mtradingdata_last.loc[0]
            print(str(i)+' : ' +'has no nan')
            

## 2.5 append the financial data into HS300alldata
HS300alldata['total_liab'] = np.nan

HS300alldata['n_cashflow_act'] = np.nan

HS300alldata['revenue'] = np.nan

HS300alldata['fcff'] = np.nan
HS300alldata['roa'] = np.nan
HS300alldata['q_opincome'] = np.nan

for i in range(HS300alldata.shape[0]):
    ## balancesheet: total liablity
    mbalancesheet = copy.deepcopy(HS300_balancesheetdata.loc[HS300_balancesheetdata.ts_code == HS300alldata.loc[i, 'Codes']])
    mbalancesheet_last = copy.deepcopy(mbalancesheet.loc[mbalancesheet.f_ann_date <= HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
    if mbalancesheet_last.shape[0] > 0 :
        m_total_liab = copy.deepcopy(mbalancesheet_last.loc[~(np.isnan(mbalancesheet_last.total_liab))].reset_index(drop=True))
        if m_total_liab.shape[0] > 0 :
            HS300alldata.loc[i, 'total_liab'] = m_total_liab.loc[0,'total_liab']
    
    ## cash flow: n_cashflow_act and free_cashflow
    mcashflow = copy.deepcopy(HS300_cashflowdata.loc[HS300_cashflowdata.ts_code == HS300alldata.loc[i, 'Codes']])
    mcashflow_last = copy.deepcopy(mcashflow.loc[mcashflow.f_ann_date <= HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
    if mcashflow_last.shape[0] > 0 :
        m_n_cashflow_act = copy.deepcopy(mcashflow_last.loc[~(np.isnan(mcashflow_last.n_cashflow_act))].reset_index(drop=True))
        if m_n_cashflow_act.shape[0] > 0:
            HS300alldata.loc[i, 'n_cashflow_act'] = m_n_cashflow_act.loc[0, 'n_cashflow_act']
        
    ## income: revenue    
    mincome = copy.deepcopy(HS300_incomedata.loc[HS300_incomedata.ts_code == HS300alldata.loc[i, 'Codes']])
    mincome_last = copy.deepcopy(mincome.loc[mincome.f_ann_date <= HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
    if mincome_last.shape[0] > 0 :
        m_income = copy.deepcopy(mincome_last.loc[~(np.isnan(mincome_last.revenue))].reset_index(drop=True))
        if m_income.shape[0] > 0 :
            HS300alldata.loc[i, 'revenue'] = m_income.loc[0,'revenue']
    
    ## financial indicator:fcff, roa and q_opincome
    mindicator = copy.deepcopy(HS300_fina_indicatordata.loc[HS300_fina_indicatordata.ts_code == HS300alldata.loc[i, 'Codes']])
    mindicator_last = copy.deepcopy(mindicator.loc[mindicator.ann_date <= HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
    if mindicator_last.shape[0] > 0 :
        m_fcff = copy.deepcopy(mindicator_last.loc[~(np.isnan(mindicator_last.fcff))].reset_index(drop=True))
        if m_fcff.shape[0] > 0:
            HS300alldata.loc[i, 'fcff'] = m_fcff.loc[0, 'fcff']
        
        m_roa = copy.deepcopy(mindicator_last.loc[~(np.isnan(mindicator_last.roa))].reset_index(drop=True))
        if m_roa.shape[0] > 0:
            HS300alldata.loc[i, 'roa'] = m_roa.loc[0, 'roa']
        
        m_q_opincome = copy.deepcopy(mindicator_last.loc[~(np.isnan(mindicator_last.q_opincome))].reset_index(drop=True))
        if m_q_opincome.shape[0] > 0:
            HS300alldata.loc[i, 'q_opincome'] = m_q_opincome.loc[0, 'q_opincome']
    print(i)
    

HS300alldata["Log_mkt_Cap"] = np.log(list(HS300alldata["total_mv"]))

HS300alldata["Cash_Over_MktCap"] = HS300alldata.fcff.div(HS300alldata.total_mv, axis=0)

HS300alldata["Rev_Over_mktCap"] = HS300alldata.revenue.div(HS300alldata.total_mv, axis=0)

HS300alldata["NOCF_Over_Debt"] = HS300alldata.n_cashflow_act.div(HS300alldata.total_liab, axis=0)


################################################################
############ Step Three: volatility for the last 60 trading days
HS300alldata['Volatility'] = np.nan

for i in range(HS300alldata.shape[0]):
    mtradingdata_vol = copy.deepcopy(HS300_tradingdata.loc[HS300_tradingdata.ts_code == HS300alldata.loc[i, 'Codes']])
    mtradingdata_vol1 = copy.deepcopy(mtradingdata_vol.loc[mtradingdata_vol.trade_date < HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    ## volatility
    if (mtradingdata_vol1.shape[0] >= vol_lag):
        HS300alldata.loc[i, 'Volatility'] = np.nanstd(mtradingdata_vol1.loc[:(vol_lag-1), 'pct_chg_hfq'])
        #print(str(i)+' : ' + 'has volatility data')
    else:
        print(str(i)+' : ' + 'has no volatility data') 


############################################################################
############ Step Four: Idiosyncratic volatility and Relative Strength Index

## input the daily trading data of HS300 index
HS300indexdata = pd.read_table('E:\\Practice in SCC\\Composite_python\\output\\HS300indexdata.txt', 
                             header='infer', encoding = None,  sep='\t')

HS300alldata['Idio_vol'] = np.nan
HS300alldata['RSI'] = np.nan

for i in range(HS300alldata.shape[0]):
    mtradingdata_ivol = copy.deepcopy(HS300_tradingdata.loc[HS300_tradingdata.ts_code == HS300alldata.loc[i, 'Codes']])
    mtradingdata_ivol1 = copy.deepcopy(mtradingdata_ivol.loc[mtradingdata_ivol.trade_date < HS300alldata.loc[i, 'Date']].reset_index(drop=True))
    
    ## Idiosyncratic volatility for the last 60 days
    if (mtradingdata_ivol1.shape[0] >= vol_lag):
        mtradingdata_ivol1_last = copy.deepcopy(mtradingdata_ivol1.loc[:(vol_lag-1)].reset_index(drop=True))
        mtradingdata_ivol1_last = copy.deepcopy(mtradingdata_ivol1_last.merge(HS300indexdata,  how='inner', left_on=['trade_date'], right_on=["trade_date"]))
        
        y = mtradingdata_ivol1_last.pct_chg_hfq
        X = mtradingdata_ivol1_last.pct_chg_y
        X = sm.add_constant(X)
        
        est=sm.OLS(y,X).fit()
        y_hat=est.predict(X)
        HS300alldata.loc[i, 'Idio_vol'] = np.nanstd(y-y_hat)
        #print(str(i)+' : ' + 'has volatility data')
    else:
        print(str(i)+' : ' + 'has no idiosyncratic volatility data')
    
    ## Relative Strength Index for the last 10 days
    if (mtradingdata_ivol1.shape[0] >= rsi_lag):
        mtradingdata_rsi_last = copy.deepcopy(mtradingdata_ivol1.loc[:(rsi_lag-1)].reset_index(drop=True))
        
        mGain = sum(mtradingdata_rsi_last.loc[mtradingdata_rsi_last.pct_chg_hfq > 0, 'pct_chg_hfq'])
        mLoss = np.abs(sum(mtradingdata_rsi_last.loc[mtradingdata_rsi_last.pct_chg_hfq < 0, 'pct_chg_hfq']))
        HS300alldata.loc[i, 'RSI'] = (mGain/(mGain+mLoss))*100
    
    print(i)
        

'''
#HS300alldata.to_csv('E:/Practice in SCC/Composite_python/output/HS300alldata_vol2.txt', sep='\t',index=False)

hs300_all_data_path = r'./data/HS300alldata_vol2.txt'
hs300_traing_mothly_path = r'./data/HS300tradingmonthly.txt'
HS300alldata = pd.read_table(hs300_all_data_path,
                             header='infer', encoding=None,  sep='\t')
# Step Four: merge the monthly trading data

# input the monthly trading data
HS300tradingmonthly = pd.read_table(hs300_traing_mothly_path,
                                    header='infer', encoding=None,  sep=',')

#HS300tradingmonthly.to_csv("E:/Practice in SCC/Composite_python/output/HS300tradingmonthly.csv", index=False, header=True)


HS300momentum = HS300tradingmonthly[['Date', 'Codes', 'last_1mon_pricechange', '12monPC_1monPC']]

HS300alldata = HS300alldata.merge(HS300momentum, how='left', left_on=["Date", 'Codes'], right_on=["Date", 'Codes'])

# Step Five: construct the multiple factors
# Size  Volatility  Momentum  Quality  Value

# 5.1  clean the missing values

date_factor = list(set(HS300alldata.Date))
date_factor.sort()

Industry_factor = list(set(HS300alldata.GicsCodes))
Industry_factor.sort()

# 5.2  define the necessary functions


def Winsor_per(ini_list, trim_perct = 5):
    """
    Given the ini_list and trim_perct, return the winsorized list with the percent parameter (trim_perct).
    
    The top and bottom trim_perct values are given values of the trimmed% and 1- trimmed% quantiles.

    Parameters
    ----------
    ini_list : a list 
    
    trim_perct : percentage of data to move from the top and bottom of the distributions

    Returns
    -------
    the winsorized list for the initial list with the trimmed percent (trim_perct)
    """
    # choose the non nan values
    ini_list_value = pd.DataFrame(ini_list).dropna()
    
    # get the percentile
    ini_list_down = np.percentile(ini_list_value[0], trim_perct)
    ini_list_up = np.percentile(ini_list_value[0], 100 - trim_perct)
    
    for i in range(len(ini_list)):
        if math.isnan(float(ini_list[i])):
            ini_list[i] = ini_list_up
        elif float(ini_list[i]) > ini_list_up:
            ini_list[i] = ini_list_up
        elif float(ini_list[i]) < ini_list_down:
            ini_list[i] = ini_list_down
    
    return ini_list


def Standard_nan(ini_list, trim_perct = 5):
    """
    Given two Parameters ini_list and trim_perct, return the standardization of the 
    initial list with the percent winsorized parameter (trim_perct).

    Parameters
    ----------
    ini_list : a list 
    
    trim_perct : percentage of data to move from the top and bottom of the distributions

    Returns
    -------
    the standardization of the winsorized ini_list with the winsorized parameter (trim_perct)  
    """
    # ini_list_winsor = stats.mstats.winsorize(np.array(ini_list), trim_perct_num)
    ini_list_winsor = Winsor_per(np.array(ini_list), trim_perct)
    ini_list_winsor_mean = np.nanmean(ini_list_winsor)
    ini_list_winsor_std = np.nanstd(ini_list_winsor)
    ini_list_winsor_stand = (ini_list_winsor - ini_list_winsor_mean)/ini_list_winsor_std
    return list(ini_list_winsor_stand)


def Winsor_std(ini_list, std_num = 3):
    ini_list_std = np.nanstd(ini_list)
    for i in range(len(ini_list)):
        if math.isnan(float(ini_list[i])):
            ini_list[i] = std_num * ini_list_std
        elif float(ini_list[i]) > std_num * ini_list_std:
            ini_list[i] = std_num * ini_list_std
        elif float(ini_list[i]) < (-1) * std_num * ini_list_std:
            ini_list[i] = (-1) * std_num * ini_list_std
    return ini_list


def ECDF(ini_list):
    ecdf = list(map(lambda x: sum(np.array(ini_list) <= x)/len(ini_list), ini_list))
    return ecdf
# 5.3  calculate the multiple factors
raw_factor = pd.DataFrame()
for i in range(len(date_factor)):
    for j in range(len(Industry_factor)):
        # extract the relative data from HS300alldata 
        logic_Date = (HS300alldata.Date == date_factor[i])
        logic_Industry = (HS300alldata.GicsCodes == Industry_factor[j])
        HS300alldata_date = copy.deepcopy(HS300alldata.loc[logic_Date & logic_Industry].reset_index(drop=True))
        # choose the industry which has at least 5 listed firms
        if HS300alldata_date.shape[0] >= listed_num:
            # print the indicator for the circulation
            print("i = ", date_factor[i], "and j =", Industry_factor[j])
            # Size Factor
            size_standard = (-1) * np.array(Standard_nan(list(HS300alldata_date.Log_mkt_Cap), trim_perct))
            size_standard_WinsorStd = Winsor_std(size_standard, std_num)
            # size_standard_WinsorStd_rank = pd.DataFrame(size_standard_WinsorStd).rank(ascending = 0)
            # size_standard_WinsorStd_ecdf = [sum(size_standard_WinsorStd <= x)/len(size_standard_WinsorStd) for x in size_standard_WinsorStd]
            size_standard_WinsorStd_ecdf = ECDF(list(size_standard_WinsorStd))
            # HS300alldata_date["Size_Factor"] = size_standard_WinsorStd_ecdf
            HS300alldata_date.insert(HS300alldata_date.shape[1], "Size_Factor", size_standard_WinsorStd_ecdf)

            # Volatility Factor
            volatility_standard = (-1) * np.array(Standard_nan(list(HS300alldata_date.Volatility), trim_perct))
            volatility_standard_WinsorStd = Winsor_std(volatility_standard, std_num)
            # volatility_standard_WinsorStd_ecdf = [sum(volatility_standard_WinsorStd <= x)/len(volatility_standard_WinsorStd) for x in volatility_standard_WinsorStd]
            volatility_standard_WinsorStd_ecdf = ECDF(list(volatility_standard_WinsorStd))
            HS300alldata_date.insert(HS300alldata_date.shape[1], "Volatility_Factor", volatility_standard_WinsorStd_ecdf)
            # Idiosyncratic volatility Factor
            Idiovolatility_standard = (-1) * np.array(Standard_nan(list(HS300alldata_date.Idio_vol), trim_perct))
            Idiovolatility_standard_WinsorStd = Winsor_std(Idiovolatility_standard, std_num)
            Idiovolatility_standard_WinsorStd_ecdf = ECDF(list(Idiovolatility_standard_WinsorStd))
            HS300alldata_date.insert(HS300alldata_date.shape[1], "IdioVolatility_Factor", Idiovolatility_standard_WinsorStd_ecdf)

            # RSI
            RSI_standard = (-1) * np.array(Standard_nan(list(HS300alldata_date.RSI), trim_perct))
            RSI_standard_WinsorStd = Winsor_std(RSI_standard, std_num)
            RSI_standard_WinsorStd_ecdf = ECDF(list(RSI_standard_WinsorStd))
            HS300alldata_date.insert(HS300alldata_date.shape[1], "RSI_Factor", RSI_standard_WinsorStd_ecdf)
            # Momentum Factor
            momentum_standard = (-1) * np.array(Standard_nan(list(HS300alldata_date["last_1mon_pricechange"]), trim_perct))
            momentum_standard_WinsorStd = Winsor_std(momentum_standard, std_num)
            # momentum_standard_WinsorStd_ecdf = [sum(momentum_standard_WinsorStd <= x)/len(momentum_standard_WinsorStd) for x in momentum_standard_WinsorStd]
            momentum_standard_WinsorStd_ecdf = ECDF(list(momentum_standard_WinsorStd))
            HS300alldata_date.insert(HS300alldata_date.shape[1], "Momentum_Factor", momentum_standard_WinsorStd_ecdf)
            # Quality Factor
            ROMkt_standard = np.array(Standard_nan(list(HS300alldata_date["Rev_Over_mktCap"]), trim_perct))
            # ROMkt_standard_ecdf = [sum(ROMkt_standard <= x)/len(ROMkt_standard) for x in ROMkt_standard]
            ROMkt_standard_ecdf =ECDF(list(ROMkt_standard))

            Accural_standard = np.array(Standard_nan(list(HS300alldata_date["q_opincome"]), trim_perct))
            # Accural_standard_ecdf = [sum(Accural_standard <= x)/len(Accural_standard) for x in Accural_standard]
            Accural_standard_ecdf =ECDF(list(Accural_standard))

            NocfOD_standard = np.array(Standard_nan(list(HS300alldata_date["NOCF_Over_Debt"]), trim_perct))
            # NocfOD_standard_ecdf = [sum(NocfOD_standard <= x)/len(NocfOD_standard) for x in NocfOD_standard]
            NocfOD_standard_ecdf = ECDF(list(NocfOD_standard))

            Quality_Factor = np.array(Quality_weight[0]) * np.array(ROMkt_standard_ecdf) + \
                             np.array(Quality_weight[1]) * np.array(Accural_standard_ecdf) + \
                             np.array(Quality_weight[2]) * np.array(NocfOD_standard_ecdf)
            
            HS300alldata_date.insert(HS300alldata_date.shape[1], "Quality_Factor", Quality_Factor)

            # Value Factor
            COMkt_standard = np.array(Standard_nan(list(HS300alldata_date["Cash_Over_MktCap"]), trim_perct))
            COMkt_standard_WinsorStd = Winsor_std(COMkt_standard, std_num)
            #COMkt_standard_WinsorStd_ecdf = [sum(COMkt_standard_WinsorStd <= x)/len(COMkt_standard_WinsorStd) for x in COMkt_standard_WinsorStd]
            COMkt_standard_WinsorStd_ecdf = ECDF(list(COMkt_standard_WinsorStd))

            ROMkt_standard_WinsorStd = Winsor_std(ROMkt_standard, std_num)
            # ROMkt_standard_WinsorStd_ecdf = [sum(ROMkt_standard_WinsorStd <= x)/len(ROMkt_standard_WinsorStd) for x in ROMkt_standard_WinsorStd]
            ROMkt_standard_WinsorStd_ecdf = ECDF(list(ROMkt_standard_WinsorStd))

            BP_standard = np.array(Standard_nan(list(HS300alldata_date["pb"]), trim_perct))
            BP_standard_WinsorStd = Winsor_std(BP_standard, std_num)
            # BP_standard_WinsorStd_ecdf = [sum(BP_standard_WinsorStd <= x)/len(BP_standard_WinsorStd) for x in BP_standard_WinsorStd]
            BP_standard_WinsorStd_ecdf =ECDF(list(BP_standard_WinsorStd))

            Value_Factor = np.array(Value_weight[0]) * np.array(COMkt_standard_WinsorStd_ecdf) + \
                           np.array(Value_weight[1]) * np.array(ROMkt_standard_WinsorStd_ecdf) + \
                           np.array(Value_weight[2]) * np.array(BP_standard_WinsorStd_ecdf)

            HS300alldata_date.insert(HS300alldata_date.shape[1], "Value_Factor", Value_Factor)

            # Overall Factor
            # HS300alldata_date["Overall_Factor"] = HS300alldata_date[factor_names].apply(lambda x: x.sum(), axis = 1)
            Overall_Factor = np.dot(HS300alldata_date[factor_names].values, np.array(Overall_weight))

            HS300alldata_date.insert(HS300alldata_date.shape[1], "Overall_Factor", Overall_Factor)

            # Update the set of factors
            # raw_factor = pd.concat([raw_factor, HS300alldata_date]).drop_duplicates()
            print(Overall_Factor)
            exit()
            if raw_factor.empty:
                raw_factor = copy.deepcopy(HS300alldata_date)
            else:
                raw_factor = pd.concat([raw_factor, HS300alldata_date], axis=0, join='inner')

# 5.4 write out the multiple factors
raw_factor.to_csv("E:/Practice in SCC/Composite_python/output/raw_factor2_5.csv", index=False, header=True)
raw_factor.to_csv('E:/Practice in SCC/Composite_python/output/raw_factor2_5.txt', sep='\t', index=False)
'''
## note the number of stocks every month
mcount = pd.DataFrame({'Date': date_factor})
mcount['Number_factor'] = np.nan
mcount['Number_alldata'] = np.nan

for i in range(mcount.shape[0]):
    raw_factor_date = raw_factor.loc[raw_factor.Date == date_factor[i]]
    mcount.loc[i, 'Number_factor'] = raw_factor_date.shape[0]
    
    alldata_date = HS300alldata.loc[HS300alldata.Date == date_factor[i]]
    mcount.loc[i, 'Number_alldata'] = alldata_date.shape[0]
    


mcount1 = pd.DataFrame({'Industry': Industry_factor})
mcount1['Number_factor'] = np.nan
mcount1['Number_alldata'] = np.nan

raw_factor1 = raw_factor.loc[raw_factor.Date == date_factor[0]].reset_index(drop=True)
alldata_date1 = HS300alldata.loc[HS300alldata.Date == date_factor[0]].reset_index(drop=True)

for i in range(mcount1.shape[0]):
    raw_factor_date = raw_factor1.loc[raw_factor1.GicsCodes == Industry_factor[i]]
    mcount1.loc[i, 'Number_factor'] = raw_factor_date.shape[0]
    
    alldata_date = alldata_date1.loc[alldata_date1.GicsCodes == Industry_factor[i]]
    mcount1.loc[i, 'Number_alldata'] = alldata_date.shape[0]
'''









