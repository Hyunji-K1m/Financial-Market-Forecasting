#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:03:06 2024

@author: kimhyunji
"""
import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import probplot
##############################################################################
# Performance evaluation functions
##############################################################################
def num_params(model):
    n_params = 0
    for p in model.params:
        if np.isnan(p):
            pass
        else:
            n_params += 1
    return n_params

def SSE(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.sum((y_test - y_pred) ** 2)

def ME(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(y_test - y_pred)

def RMSE(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.sqrt(np.mean((y_test - y_pred) ** 2))

def MPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean((y_test - y_pred) / y_test) * 100

def MAPE(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def AIC(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sse = np.sum((y_test - y_pred) ** 2)
    k = num_params(model)  
    return T * np.log(sse / T) + 2 * k

def SBC(y_test, y_pred, T, model):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    sse = np.sum((y_test - y_pred) ** 2)
    k = num_params(model)
    return T * np.log(sse / T) + k * np.log(T)

def APC(y_true, y_pred, T, model):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sse = np.sum((y_true - y_pred) ** 2)
    k = num_params(model) 
    return ((T + k) / (T - k)) * sse / T

def ADJ_R2(y_true, y_pred, T, model):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    k = num_params(model)
    r2 = 1 - sse/sst
    return 1 - ((T - 1) / (T - k)) * (1 - r2)

##############################################################################
# K54D data set
##############################################################################
weekly_earnings_path = '/Users/kimhyunji/Desktop/Semester 2/Forecasting/Average weekly earnings data set.xlsx'
weekly_earnings = pd.read_excel(weekly_earnings_path, usecols=[0,75])
month_earnings = weekly_earnings.iloc[570:858].set_index(weekly_earnings.columns[0])

def month_to_date(month_string):
    if isinstance(month_string, int):
        month_string = str(month_string)
    year, month_abbr = month_string.split(' ')
    month_number = datetime.strptime(month_abbr, '%b').month
    return pd.Timestamp(year=int(year), month=month_number, day=15)

month_earnings.index = month_earnings.index.astype(str)  
month_earnings.index = month_earnings.index.map(month_to_date)
month_earnings.columns = ['K54D']
month_earnings = month_earnings.astype(float)

########################
# transform log
########################
month_earnings['K54D_Log'] = np.log(month_earnings['K54D'] + 1)

########################
# ADF, ACF, PACF
########################
adf_result = adfuller(month_earnings['K54D_Log'])
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
for key, value in adf_result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

plt.figure(figsize=(12, 6))
plot_acf(month_earnings['K54D_Log'], ax=plt.gca())
plt.title('Autocorrelation Function')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(month_earnings['K54D_Log'], ax=plt.gca())
plt.title('Partial Autocorrelation Function')
plt.show()

########################
# Difference, ACF, PACF
########################
diff_month_earnings = month_earnings['K54D_Log'].diff().dropna()
adf_diff = adfuller(diff_month_earnings)
print(f'ADF Statistic: {adf_diff[0]}')
print(f'p-value: {adf_diff[1]}')
for key, value in adf_diff[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
diff_seasonal_month_earnings = diff_month_earnings.diff(12).dropna()
adf_seasonal = adfuller(diff_seasonal_month_earnings)
print(f'ADF Statistic: {adf_seasonal[0]}')
print(f'p-value: {adf_seasonal[1]}')
for key, value in adf_seasonal[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

plt.figure(figsize=(12, 6))
plot_acf(diff_seasonal_month_earnings, ax=plt.gca())
plt.title('Seasonally Differenced Autocorrelation Function')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(diff_seasonal_month_earnings, ax=plt.gca())
plt.title('Seasonally Differenced Partial Autocorrelation Function')
plt.show()

########################
# Find optimal parameters
# ARIMA(3,1,1)(2,0,1)[12]
########################
auto_arima_model = auto_arima(diff_seasonal_month_earnings,d=1, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore", trace=True)
print(auto_arima_model.summary())

auto_arima_model.plot_diagnostics(figsize=(16,8))
plt.show()

########################
# Fit an ARIMA model
########################
train_earnings = month_earnings[month_earnings.index < '2023-01-01']['K54D_Log']
test_earnings = month_earnings[(month_earnings.index >= '2023-01-01') & (month_earnings.index < '2024-01-01')]['K54D_Log']
p, d, q = 3, 1, 1
P, D, Q, s = 2, 0, 1, 12
model = SARIMAX(train_earnings, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit()

forecast = model_fit.get_forecast(steps=len(test_earnings))
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

T = len(test_earnings)
sse = SSE(test_earnings, mean_forecast)
mse = MSE(test_earnings, mean_forecast)
rmse = RMSE(test_earnings, mean_forecast)
me = ME(test_earnings, mean_forecast)
mae = MAE(test_earnings, mean_forecast)
mpe = MPE(test_earnings, mean_forecast)
mape = MAPE(test_earnings, mean_forecast)
aic = AIC(test_earnings, mean_forecast, T, model_fit)
sbc = SBC(test_earnings, mean_forecast, T, model_fit)
apc = APC(test_earnings, mean_forecast, T, model_fit)
adj_r2 = ADJ_R2(test_earnings, mean_forecast, T, model_fit)

df_earnings_2023 = pd.DataFrame({
    'SARIMA': [sse, mse, rmse, me, mae, mpe, mape, aic, sbc, apc, adj_r2]
}, index=['SSE', 'MSE', 'RMSE', 'ME', 'MAE', 'MPE', 'MAPE', 'AIC', 'SBC', 'APC', 'Adj_R2'])

print(df_earnings_2023)

f_scale_earnings_2023 = np.exp(mean_forecast) - 1
actual_earnings = np.exp(test_earnings) - 1
conf_scaled=np.exp(conf_int)-1
plt.figure(figsize=(10, 6))
plt.plot(month_earnings.index, np.exp(month_earnings['K54D_Log']) - 1, label='Actual')
plt.plot(test_earnings.index, f_scale_earnings_2023, label='Forecast')
plt.fill_between(test_earnings.index, conf_scaled.iloc[:, 0], conf_scaled.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('K54D_Forecast 2023')
plt.show()

########################
# K54D_2023_residuals
########################
actual_earnings = actual_earnings.reset_index(drop=True)
f_scale_earnings_2023 = f_scale_earnings_2023.reset_index(drop=True)

residuals_earnings = actual_earnings - f_scale_earnings_2023

plot_acf(residuals_earnings, ax=plt.gca())
plt.title('ACF of Residuals_K54D')
plt.show()

probplot(residuals_earnings, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_K54D')
plt.show()

########################
# Forecasting K54D_2024 
########################
forecast_2024 = model_fit.get_forecast(steps=12)
forecast_mean_2024 = forecast_2024.predicted_mean
conf_int_2024 = forecast_2024.conf_int()
fit_original_earnings = np.exp(forecast_mean_2024) - 1
print(fit_original_earnings)

plt.figure(figsize=(10,6))
plt.plot(month_earnings.index, month_earnings['K54D'], label='Actual')
plt.plot(pd.date_range(start='1/1/2024', end='31/12/2024', freq='M'), fit_original_earnings, label='Forecast')
plt.legend()
plt.title('K54D_Forecast 2024')
plt.show()
