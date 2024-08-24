#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:22:41 2024

@author: kimhyunji
"""
import pandas as pd
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.tsa.stattools import acf, q_stat
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox

##############################################################################
#average weekly earnings data set
##############################################################################
weekly_earnings_path = '/Users/kimhyunji/Desktop/Semester 2/Forecasting/Average weekly earnings data set.xlsx'
weekly_earnings = pd.read_excel(weekly_earnings_path, usecols=[0,75])
year_earnings = weekly_earnings.iloc[6:30].set_index(weekly_earnings.columns[0])
Q14_earnings = weekly_earnings.iloc[30:126].set_index(weekly_earnings.columns[0])
month_earnings = weekly_earnings.iloc[570:858].set_index(weekly_earnings.columns[0])

##############################################################################
# retail sales data set
##############################################################################
retail_sales_path = '/Users/kimhyunji/Desktop/Semester 2/Forecasting/From retail sales time series data.xlsx'
retail_sales = pd.read_excel(retail_sales_path, usecols=[0,34])
year_retail= retail_sales.iloc[18:42].set_index(retail_sales.columns[0])
Q14_retail= retail_sales.iloc[90:186].set_index(retail_sales.columns[0])
month_retail= retail_sales.iloc[330:618].set_index(retail_sales.columns[0])

##############################################################################
# production_data_set
##############################################################################
production_path='/Users/kimhyunji/Desktop/Semester 2/Forecasting/From the index of production data set.xlsx'
production=pd.read_excel(production_path, usecols=[0,153])
year_production= production.iloc[58:82].set_index(production.columns[0])
Q14_production= production.iloc[290:386].set_index(production.columns[0])
month_production= production.iloc[1010:1298].set_index(production.columns[0])

##############################################################################
# manufacturing data set
##############################################################################
manu_path='/Users/kimhyunji/Desktop/Semester 2/Forecasting/series-150224.xls'
manu=pd.read_excel(manu_path, usecols=[0,1])
year_manu= manu.iloc[9:33].set_index(manu.columns[0])
Q14_manu= manu.iloc[41:137].set_index(manu.columns[0])
month_manu= manu.iloc[161:449].set_index(manu.columns[0])

##############################################################################
#quarter function
##############################################################################
def quarter_to_date(quarter_string):
    year, quarter = quarter_string.split(' ')
    first_month_of_quarter = (int(quarter[-1]) - 1) * 3 + 1
    return f"{year}-{first_month_of_quarter:02d}-01"

########################
# quarter_time series_all data
########################
"""
Q14_earnings.index = Q14_earnings.index.map(quarter_to_date)
Q14_earnings.index = pd.to_datetime(Q14_earnings.index)
Q14_earnings.columns = ['Earnings']
Q14_earnings = Q14_earnings.astype(float)

Q14_retail.index = Q14_retail.index.map(quarter_to_date)
Q14_retail.index = pd.to_datetime(Q14_retail.index)
Q14_retail.columns = ['Retail']
Q14_retail = Q14_retail.astype(float)

Q14_production.index = Q14_production.index.map(quarter_to_date)
Q14_production.index = pd.to_datetime(Q14_production.index)
Q14_production.columns = ['Production']
Q14_production = Q14_production.astype(float)

Q14_manu.index = Q14_manu.index.map(quarter_to_date)
Q14_manu.index = pd.to_datetime(Q14_manu.index)
Q14_manu.columns = ['Manufacturing']
Q14_manufacturing = Q14_manu.astype(float)

Q14_seasonal_earnings = seasonal_decompose(Q14_earnings['Earnings'], model='additive',period=4)
plt.rcParams['figure.figsize'] = [8, 6]
Q14_seasonal_earnings.plot()
plt.show()

Q14_seasonal_retail = seasonal_decompose(Q14_retail['Retail'], model='additive',period=4)
Q14_seasonal_retail.plot()
plt.show()

Q14_seasonal_production = seasonal_decompose(Q14_production['Production'], model='additive',period=4)
Q14_seasonal_production.plot()
plt.show()

Q14_seasonal_manu = seasonal_decompose(Q14_manu['Manufacturing'], model='additive',period=4)
Q14_seasonal_manu.plot()
plt.show()
"""
##############################################################################
# month functions
##############################################################################
def month_to_date(month_string):
    if isinstance(month_string, int):
        month_string = str(month_string)
    year, month_abbr = month_string.split(' ')
    month_number = datetime.strptime(month_abbr, '%b').month
    return pd.Timestamp(year=int(year), month=month_number, day=15)

def find_outliers(residuals, std_dev_factor=2):
    r_mean=residuals.mean()
    r_std=residuals.std()
    outliers=(residuals>r_mean + std_dev_factor * r_std) | (residuals<r_mean - std_dev_factor * r_std)
    outliers_dates = residuals[outliers].index.tolist()
    return outliers_dates

##############################################################################
# Performance evaluation functions
##############################################################################
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def num_params(model):
    n_params = 0

    for p in list(model.params.values()):
        if isinstance(p, np.ndarray):
            n_params += len(p)
            #print(p)
        elif p in [np.nan, False, None]:
            pass
        elif np.isnan(float(p)):
            pass
        else:
            n_params += 1
            #print(p)
    
    return n_params

def SSE(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.sum((y_test - y_pred)**2)

def ME(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(y_test - y_pred)

def RMSE(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.sqrt(np.mean((y_test - y_pred)**2))   
    #return np.sqrt(MSE(y_test - y_pred))

def MPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean((y_test - y_pred) / y_test) * 100

def MAPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def AIC(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sse = np.sum((y_test - y_pred)**2)
    #T = len(y_train) # number of observations
    k = num_params(model) # number of parameters
    return T * np.log(sse/T) + 2*k

def SBC(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sse = np.sum((y_test - y_pred)**2)
    #T = len(y_train) # number of observations
    k = num_params(model) # number of parameters
    return T * np.log(sse/T) + k * np.log(T)

def APC(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sse = np.sum((y_test - y_pred)**2)
    #T = len(y_train) # number of observations
    k = num_params(model) # number of parameters
    return ((T+k)/(T-k)) * sse / T

def ADJ_R2(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sst = np.sum((y_test - np.mean(y_test))**2)
    sse = np.sum((y_test - y_pred)**2)
    #T = len(y_train) # number of observations
    k = num_params(model) # number of parameters
    r2 = 1 - sse/sst
    return 1 - ((T - 1)/(T - k)) * (1 - r2)

def eval_all(y_test, y_pred, T, model):
    sse = SSE(y_test, y_pred)
    mse = MSE(y_test, y_pred)
    rmse = RMSE(y_test, y_pred)
    me = ME(y_test, y_pred)
    mae = MAE(y_test, y_pred)
    mpe = MPE(y_test, y_pred)
    mape = MAPE(y_test, y_pred)
    aic = AIC(y_test, y_pred, T, model)
    sbc = SBC(y_test, y_pred, T, model)
    apc = APC(y_test, y_pred, T, model)
    adj_r2 = ADJ_R2(y_test, y_pred, T, model)
    
    return [sse, mse, rmse, me, mae, mpe, mape, aic, sbc, apc, adj_r2]

########################
# month_time series_all data
########################
month_earnings.index = month_earnings.index.astype(str)  
month_earnings.index = month_earnings.index.map(month_to_date)
month_earnings.columns = ['K54D']
month_earnings = month_earnings.astype(float)

month_retail.index = month_retail.index.astype(str)  
month_retail.index = month_retail.index.map(month_to_date)
month_retail.columns = ['EAFV']
month_retail = month_retail.astype(float)

month_production.index = month_production.index.astype(str)  
month_production.index = month_production.index.map(month_to_date)
month_production.columns = ['K226']
month_production = month_production.astype(float)

month_manu.index = month_manu.index.astype(str)  
month_manu.index = month_manu.index.map(month_to_date)
month_manu.columns = ['JQ2J']
month_manu = month_manu.astype(float)

month_seasonal_earnings = seasonal_decompose(month_earnings['K54D'], model='additive', period=12)
plt.rcParams['figure.figsize'] = [8, 6]
month_seasonal_earnings.plot()
plt.show()

month_seasonal_retail = seasonal_decompose(month_retail['EAFV'], model='additive', period=12)
month_seasonal_retail.plot()
plt.show()

month_seasonal_production = seasonal_decompose(month_production['K226'], model='additive', period=12)
month_seasonal_production.plot()
plt.show()

month_seasonal_manu = seasonal_decompose(month_manu['JQ2J'], model='additive', period=12)
month_seasonal_manu.plot()
plt.show()

########################
# outliers
########################
earnings_residuals = month_seasonal_earnings.resid.dropna()
outliers_dates=find_outliers(earnings_residuals)
for date in outliers_dates:
    print(f"K54D Outlier found at: {date.strftime('%Y-%m')}")
    
retail_residuals = month_seasonal_retail.resid.dropna()
outliers_dates=find_outliers(retail_residuals)
for date in outliers_dates:
    print(f"EAFV Outlier found at: {date.strftime('%Y-%m')}")
    
production_residuals = month_seasonal_production.resid.dropna()
outliers_dates=find_outliers(production_residuals)
for date in outliers_dates:
    print(f"K226 Outlier found at: {date.strftime('%Y-%m')}")
    
manu_residuals = month_seasonal_manu.resid.dropna()
outliers_dates=find_outliers(manu_residuals)
for date in outliers_dates:
    print(f"JQ2J Outlier found at: {date.strftime('%Y-%m')}")

########################
# transform log
########################
month_earnings['K54D_Log'] = np.log(month_earnings['K54D'] + 1)
"""
month_seasonal_earnings_log = seasonal_decompose(month_earnings['K54D_Log'], model='additive', period=12)
month_seasonal_earnings_log.plot()
plt.show()
"""
month_retail['EAFV_Log'] = np.log(month_retail['EAFV'] + 1)
month_production['K226_Log'] = np.log(month_production['K226'] + 1)
month_manu['JQ2J_Log'] = np.log(month_manu['JQ2J'] + 1)

########################
# train-test set
########################
train_earnings = month_earnings[month_earnings.index < '2023-01-01']['K54D_Log']
test_earnings = month_earnings[(month_earnings.index >= '2023-01-01') & (month_earnings.index < '2024-01-01')]['K54D_Log']

train_retail = month_retail[month_retail.index < '2023-01-01']['EAFV_Log']
test_retail = month_retail[(month_retail.index >= '2023-01-01') & (month_retail.index < '2024-01-01')]['EAFV_Log']

train_production = month_production[month_production.index < '2023-01-01']['K226_Log']
test_production = month_production[(month_production.index >= '2023-01-01') & (month_production.index < '2024-01-01')]['K226_Log']

train_manu = month_manu[month_manu.index < '2023-01-01']['JQ2J_Log']
test_manu = month_manu[(month_manu.index >= '2023-01-01') & (month_manu.index < '2024-01-01')]['JQ2J_Log']

########################
# K54D_2023
########################
fit1_earnings = SimpleExpSmoothing(train_earnings, initialization_method="estimated").fit()
fit2_earnings = Holt(train_earnings, initialization_method="estimated").fit()
fit3_earnings = Holt(train_earnings, exponential=True, initialization_method="estimated").fit()
fit4_earnings = Holt(train_earnings, damped_trend=True, initialization_method="estimated").fit()
fit5_earnings = Holt(train_earnings, exponential=True, damped_trend=True, initialization_method="estimated").fit()
fit6_earnings = HWES(train_earnings, seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit7_earnings = HWES(train_earnings, seasonal_periods=12, trend='add', seasonal='mul',damped_trend=True).fit(optimized=True)

f1_earnings = fit1_earnings.forecast(len(test_earnings))
f2_earnings = fit2_earnings.forecast(len(test_earnings))
f3_earnings = fit3_earnings.forecast(len(test_earnings))
f4_earnings = fit4_earnings.forecast(len(test_earnings))
f5_earnings = fit5_earnings.forecast(len(test_earnings))
f6_earnings = fit6_earnings.forecast(len(test_earnings))
f7_earnings = fit7_earnings.forecast(len(test_earnings))
#actual_earnings = np.exp(test_earnings) - 1
T = train_earnings.shape[0]

df_earnings_2023 = pd.DataFrame(
    {'SES': eval_all(test_earnings, f1_earnings, T, fit1_earnings), 
    "Holt's": eval_all(test_earnings, f2_earnings, T, fit2_earnings), 
    'Exponential': eval_all(test_earnings, f3_earnings, T, fit3_earnings), 
    'Trend_Add': eval_all(test_earnings, f4_earnings, T, fit4_earnings), 
    'Trend_Mult': eval_all(test_earnings, f5_earnings, T, fit5_earnings), 
    'Trend_Season_Add': eval_all(test_earnings, f6_earnings, T, fit6_earnings),
    'Trend_Season_Mul': eval_all(test_earnings, f7_earnings, T, fit7_earnings)}
    , index=['SSE', 'MSE', 'RMSE', 'ME', 'MAE', 'MPE', 'MAPE', 'AIC', 'SBC', 'APC', 'Adj_R2'])
pd.set_option('display.max_columns', None)
print(df_earnings_2023)
"""
eval_all_df.loc['MAPE', :].plot(kind='barh', figsize=[8, 6])
plt.title('Mean Absolute Percentage Error (MAPE)', fontsize=16)
plt.show()
"""
print(fit7_earnings.summary())

f_scale_earnings_2023 = np.exp(f7_earnings) - 1
actual_earnings = np.exp(test_earnings) - 1
plt.figure(figsize=(10, 6))
plt.plot(month_earnings.index, np.exp(month_earnings['K54D_Log']) - 1, label='Actual')
plt.plot(test_earnings.index, f_scale_earnings_2023, label='Forecast')
plt.legend()
plt.title('K54D_Forecast 2023')
plt.show()

########################
# K54D_2023_residuals
########################
actual_earnings = actual_earnings.reset_index(drop=True)
f_scale_earnings_2023 = f_scale_earnings_2023.reset_index(drop=True)

residuals_earnings = actual_earnings - f_scale_earnings_2023

plot_acf(residuals_earnings,lags=11)
plt.title('ACF of Residuals_K54D')
plt.show()

probplot(residuals_earnings, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_K54D')
plt.show()

lb_test = acorr_ljungbox(residuals_earnings, lags=[10], return_df=True)
print(lb_test)

########################
# EAFV_2023
########################
fit1_retail = SimpleExpSmoothing(train_retail, initialization_method="estimated").fit()
fit2_retail = Holt(train_retail, initialization_method="estimated").fit()
fit3_retail = Holt(train_retail, exponential=True, initialization_method="estimated").fit()
fit4_retail = Holt(train_retail, damped_trend=True, initialization_method="estimated").fit()
fit5_retail = Holt(train_retail, exponential=True, damped_trend=True, initialization_method="estimated").fit()
fit6_retail = HWES(train_retail, seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit7_retail = HWES(train_retail, seasonal_periods=12, trend='add', seasonal='mul',damped_trend=True).fit(optimized=True)

f1_retail = fit1_retail.forecast(len(test_retail))
f2_retail = fit2_retail.forecast(len(test_retail))
f3_retail = fit3_retail.forecast(len(test_retail))
f4_retail = fit4_retail.forecast(len(test_retail))
f5_retail = fit5_retail.forecast(len(test_retail))
f6_retail = fit6_retail.forecast(len(test_retail))
f7_retail = fit7_retail.forecast(len(test_retail))

T = train_retail.shape[0]

df_retail_2023 = pd.DataFrame(
    {'SES': eval_all(test_retail, f1_retail, T, fit1_retail), 
    "Holt's": eval_all(test_retail, f2_retail, T, fit2_retail), 
    'Exponential': eval_all(test_retail, f3_retail, T, fit3_retail), 
    'Trend_Add': eval_all(test_retail, f4_retail, T, fit4_retail), 
    'Trend_Mult': eval_all(test_retail, f5_retail, T, fit5_retail), 
    'Trend_Season_Add': eval_all(test_retail, f6_retail, T, fit6_retail),
    'Trend_Season_Mul': eval_all(test_retail, f7_retail, T, fit7_retail)}
    , index=['SSE', 'MSE', 'RMSE', 'ME', 'MAE', 'MPE', 'MAPE', 'AIC', 'SBC', 'APC', 'Adj_R2'])
pd.set_option('display.max_columns', None)
print(df_retail_2023)

print(fit6_retail.summary())

actual_retail=np.exp(test_retail)-1
f_scale_retail_2023 = np.exp(f6_retail) - 1
actual_retail = np.exp(test_retail) - 1
plt.figure(figsize=(10, 6))
plt.plot(month_retail.index, np.exp(month_retail['EAFV_Log']) - 1, label='Actual')
plt.plot(test_retail.index, f_scale_retail_2023, label='Forecast')
plt.legend()
plt.title('EAFV_Forecast 2023')
plt.show()

########################
# EAFV_2023_residuals
########################
actual_retail = actual_retail.reset_index(drop=True)
f_scale_retail_2023 = f_scale_retail_2023.reset_index(drop=True)

residuals_retail = actual_retail - f_scale_retail_2023

plot_acf(residuals_retail,lags=11)
plt.title('ACF of Residuals_EAFV')
plt.show()

probplot(residuals_retail, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_EAFV')
plt.show()

lb_test = acorr_ljungbox(residuals_retail, lags=[10], return_df=True)
print(lb_test)

########################
# K226_2023
########################
fit1_production = SimpleExpSmoothing(train_production, initialization_method="estimated").fit()
fit2_production = Holt(train_production, initialization_method="estimated").fit()
fit3_production = Holt(train_production, exponential=True, initialization_method="estimated").fit()
fit4_production = Holt(train_production, damped_trend=True, initialization_method="estimated").fit()
fit5_production = Holt(train_production, exponential=True, damped_trend=True, initialization_method="estimated").fit()
fit6_production = HWES(train_production, seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit7_production = HWES(train_production, seasonal_periods=12, trend='mul', seasonal='add').fit(optimized=True)

f1_production = fit1_production.forecast(len(test_production))
f2_production = fit2_production.forecast(len(test_production))
f3_production = fit3_production.forecast(len(test_production))
f4_production = fit4_production.forecast(len(test_production))
f5_production = fit5_production.forecast(len(test_production))
f6_production = fit6_production.forecast(len(test_production))
f7_production = fit7_production.forecast(len(test_production))

T = train_production.shape[0]

df_production_2023 = pd.DataFrame(
    {'SES': eval_all(test_production, f1_production, T, fit1_production), 
    "Holt's": eval_all(test_production, f2_production, T, fit2_production), 
    'Exponential': eval_all(test_production, f3_production, T, fit3_production), 
    'Trend_Add': eval_all(test_production, f4_production, T, fit4_production), 
    'Trend_Mult': eval_all(test_production, f5_production, T, fit5_production), 
    'Trend_Season_Add': eval_all(test_production, f6_production, T, fit6_production),
    'Trend_Season_Mul': eval_all(test_production, f7_production, T, fit7_production)}
    , index=['SSE', 'MSE', 'RMSE', 'ME', 'MAE', 'MPE', 'MAPE', 'AIC', 'SBC', 'APC', 'Adj_R2'])
pd.set_option('display.max_columns', None)
print(df_production_2023)

print(fit6_production.summary())

actual_production=np.exp(test_production)-1
f_scale_production_2023 = np.exp(f7_production) - 1
plt.figure(figsize=(10, 6))
plt.plot(month_production.index, np.exp(month_production['K226_Log']) - 1, label='Actual')
plt.plot(test_production.index, f_scale_production_2023, label='Forecast')
plt.legend()
plt.title('K226_Forecast 2023')
plt.show()

########################
# K226_2023_residuals
########################
actual_production = actual_production.reset_index(drop=True)
f_scale_production_2023 = f_scale_production_2023.reset_index(drop=True)

residuals_production = actual_production - f_scale_production_2023

plot_acf(residuals_production,lags=11)
plt.title('ACF of Residuals_K226')
plt.show()

probplot(residuals_production, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_K226')
plt.show()

lb_test = acorr_ljungbox(residuals_production, lags=[10], return_df=True)
print(lb_test)

########################
# JQ2J_2023
########################
fit1_manu = SimpleExpSmoothing(train_manu, initialization_method="estimated").fit()
fit2_manu = Holt(train_manu, initialization_method="estimated").fit()
fit3_manu = Holt(train_manu, exponential=True, initialization_method="estimated").fit()
fit4_manu = Holt(train_manu, damped_trend=True, initialization_method="estimated").fit()
fit5_manu = Holt(train_manu, exponential=True, damped_trend=True, initialization_method="estimated").fit()
fit6_manu = HWES(train_manu, seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit7_manu = HWES(train_manu, seasonal_periods=12, trend='mul', seasonal='add').fit(optimized=True)

f1_manu = fit1_manu.forecast(len(test_manu))
f2_manu = fit2_manu.forecast(len(test_manu))
f3_manu = fit3_manu.forecast(len(test_manu))
f4_manu = fit4_manu.forecast(len(test_manu))
f5_manu = fit5_manu.forecast(len(test_manu))
f6_manu = fit6_manu.forecast(len(test_manu))
f7_manu = fit7_manu.forecast(len(test_manu))

T = train_manu.shape[0]

df_manu_2023 = pd.DataFrame(
    {'SES': eval_all(test_manu, f1_manu, T, fit1_manu), 
    "Holt's": eval_all(test_manu, f2_manu, T, fit2_manu), 
    'Exponential': eval_all(test_manu, f3_manu, T, fit3_manu), 
    'Trend_Add': eval_all(test_manu, f4_manu, T, fit4_manu), 
    'Trend_Mult': eval_all(test_manu, f5_manu, T, fit5_manu), 
    'Trend_Season_Add': eval_all(test_manu, f6_manu, T, fit6_manu),
    'Trend_Season_Mul': eval_all(test_manu, f7_manu, T, fit7_manu)}
    , index=['SSE', 'MSE', 'RMSE', 'ME', 'MAE', 'MPE', 'MAPE', 'AIC', 'SBC', 'APC', 'Adj_R2'])
pd.set_option('display.max_columns', None)
print(df_manu_2023)

print(fit6_production.summary())

actual_manu= np.exp(test_manu) -1
f_scale_manu_2023 = np.exp(f6_manu) - 1
plt.figure(figsize=(10, 6))
plt.plot(month_manu.index, np.exp(month_manu['JQ2J_Log']) - 1, label='Actual')
plt.plot(test_manu.index, f_scale_manu_2023, label='Forecast')
plt.legend()
plt.title('JQ2J_Forecast 2023')
plt.show()

########################
# JQ2J_2023_residuals
########################
actual_manu = actual_manu.reset_index(drop=True)
f_scale_manu_2023 = f_scale_manu_2023.reset_index(drop=True)

residuals_manu = actual_manu - f_scale_manu_2023

plot_acf(residuals_manu,lags=11)
plt.title('ACF of Residuals_JQ2J')
plt.show()

probplot(residuals_manu, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_JQ2J')
plt.show()

lb_test = acorr_ljungbox(residuals_manu, lags=[10], return_df=True)
print(lb_test)

########################
# Forecasting K54D_2024 
########################
fit_earnings = HWES(month_earnings['K54D_Log'], seasonal_periods=12, trend='add', seasonal='mul',damped_trend=True).fit(optimized=True)
fit_earnings = fit_earnings.forecast(12)
fit_original_earnings = np.exp(fit_earnings) - 1
print(fit_original_earnings)

plt.figure(figsize=(10,6))
plt.plot(month_earnings.index, month_earnings['K54D'], label='Actual')
plt.plot(pd.date_range(start='1/1/2024', end='31/12/2024', freq='M'), fit_original_earnings, label='Forecast')
plt.legend()
plt.title('K54D_Forecast 2024')
plt.show()

########################
# Forecasting EAFV_2024 
########################
fit_retail = HWES(month_retail['EAFV_Log'], seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit_retail = fit_retail.forecast(12)
fit_original_retail = np.exp(fit_retail) - 1
print(fit_original_retail)

plt.figure(figsize=(10,6))
plt.plot(month_retail.index, month_retail['EAFV'], label='Actual')
plt.plot(pd.date_range(start='1/1/2024', end='31/12/2024', freq='M'), fit_original_retail, label='Forecast')
plt.legend()
plt.title('EAFV_Forecast 2024')
plt.show()

########################
# Forecasting K226_2024 
########################
fit_production = HWES(month_production['K226_Log'], seasonal_periods=12, trend='add', seasonal='mul',damped_trend=True).fit(optimized=True)
fit_production = fit_production.forecast(12)
fit_original_production = np.exp(fit_production) - 1
print(fit_original_production)

plt.figure(figsize=(10,6))
plt.plot(month_production.index, month_production['K226'], label='Actual')
plt.plot(pd.date_range(start='1/1/2024', end='31/12/2024', freq='M'), fit_original_production, label='Forecast')
plt.legend()
plt.title('K226_Forecast 2024')
plt.show()

########################
# Forecasting JQ2J_2024 
########################
fit_manu = HWES(month_manu['JQ2J_Log'], seasonal='add', trend='add', seasonal_periods=12, damped_trend=True).fit(optimized=True)
fit_manu = fit_manu.forecast(12)
fit_original_manu = np.exp(fit_manu) - 1
print(fit_original_manu)

plt.figure(figsize=(10,6))
plt.plot(month_manu.index, month_manu['JQ2J'], label='Actual')
plt.plot(pd.date_range(start='1/1/2024', end='31/12/2024', freq='M'), fit_original_manu, label='Forecast')
plt.legend()
plt.title('JQ2J_Forecast 2024')
plt.show()