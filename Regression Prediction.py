#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kimhyunji
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import combinations
import statsmodels.api as sm
from scipy.stats import probplot
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

##############################################################################
#function
##############################################################################
def month_to_date(month_string):
    if isinstance(month_string, int):
        month_string = str(month_string)
    year, month_abbr = month_string.split(' ')
    month_number = datetime.strptime(month_abbr, '%b').month
    return pd.Timestamp(year=int(year), month=month_number, day=1)

def month_to_date2(month_string):
    if isinstance(month_string, int):  
        month_string = str(month_string)
    day, month, year = month_string.split('/')  
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def calculate_vif(df, variables):
    vif_df = pd.DataFrame()
    vif_df["variable"] = variables
    vif_df["VIF"] = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]
    return vif_df

def convert_to_float(value):
    return float(value.replace(',', ''))

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
#data set
##############################################################################
weekly_earnings_path = file_path
weekly_earnings = pd.read_excel(weekly_earnings_path, usecols=[0,75])
month_earnings = weekly_earnings.iloc[583:870].set_index(weekly_earnings.columns[0])
month_earnings.index = month_earnings.index.astype(str)  
month_earnings.index = month_earnings.index.map(month_to_date)
month_earnings.columns = ['K54D']
month_earnings = month_earnings.astype(float)

retail_sales_path = file_path
retail_sales = pd.read_excel(retail_sales_path, usecols=[0,34])
month_retail= retail_sales.iloc[343:630].set_index(retail_sales.columns[0])
month_retail.index = month_retail.index.astype(str)  
month_retail.index = month_retail.index.map(month_to_date)
month_retail.columns = ['EAFV']
month_retail = month_retail.astype(float)

production_path= file_path
production=pd.read_excel(production_path, usecols=[0,153])
month_production= production.iloc[1023:1310].set_index(production.columns[0])
month_production.index = month_production.index.astype(str)  
month_production.index = month_production.index.map(month_to_date)
month_production.columns = ['K226']
month_production = month_production.astype(float)

manu_path= file_path
manu=pd.read_excel(manu_path, usecols=[0,1])
month_manu= manu.iloc[174:461].set_index(manu.columns[0])
month_manu.index = month_manu.index.astype(str)  
month_manu.index = month_manu.index.map(month_to_date)
month_manu.columns = ['JQ2J']
month_manu = month_manu.astype(float)
data = pd.concat([month_earnings['K54D'],month_retail['EAFV'], month_production['K226'], 
                   month_manu['JQ2J'],], axis=1, join='inner')
x_columns = ['K54D', 'EAFV', 'K226', 'JQ2J']

ftse_path= file_path
ftse=pd.read_csv(ftse_path,usecols=[0,2,3,4])
month_ftse= ftse.iloc[2:277].set_index(ftse.columns[0])
month_ftse = month_ftse.sort_index(ascending=True)
month_ftse.index = month_ftse.index.astype(str)
month_ftse.index = month_ftse.index.map(month_to_date2)
month_ftse.columns = ['FTSE Open Price','FTSE High','FTSE Low']
month_ftse['FTSE Open Price'] = month_ftse['FTSE Open Price'].apply(convert_to_float)
month_ftse['FTSE High'] = month_ftse['FTSE High'].apply(convert_to_float)
month_ftse['FTSE Low'] = month_ftse['FTSE Low'].apply(convert_to_float)
month_ftse = month_ftse.astype(float)

x = data
scaler = StandardScaler()
original_index = data.index
x_scaled = scaler.fit_transform(x)
data = pd.DataFrame(x_scaled, index=original_index, columns=x_columns)
##############################################################################
# VIF
##############################################################################
vif = calculate_vif(data, x_columns)
print("VIF for variables:\n", vif)

correlation_matrix = data.corr()

##############################################################################
# Correlation
##############################################################################
cor = data[x_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(cor, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

##############################################################################
# Multiple Regression (backward)
##############################################################################
for col1, col2 in combinations(x_columns, 2):
    interaction_term = col1 + '_X_' + col2
    data[interaction_term] = data[col1] * data[col2]

for col1, col2, col3 in combinations(x_columns, 3):
    interaction_term = col1 + '_X_' + col2 + '_X_' + col3
    data[interaction_term] = data[col1] * data[col2] * data[col3]

if len(x_columns) == 4:
    interaction_term = '_X_'.join(x_columns)
    data[interaction_term] = data[x_columns[0]] * data[x_columns[1]] * data[x_columns[2]] * data[x_columns[3]]

extended_x = x_columns.copy()
extended_x += [col1 + '_X_' + col2 for col1, col2 in combinations(x_columns, 2)]
extended_x += [col1 + '_X_' + col2 + '_X_' + col3 for col1, col2, col3 in combinations(x_columns, 3)]
if len(x_columns) == 4:
    extended_x.append('_X_'.join(x_columns))
print("Extended independent variables:", extended_x)

train_x = data.loc[data.index < '2023-01-01', extended_x]
test_x = data.loc[(data.index >= '2023-01-01') & (data.index < '2024-01-01'), extended_x]
train_y = month_ftse[month_ftse.index < '2023-01-01']['FTSE Open Price']
test_y = month_ftse[(month_ftse.index >= '2023-01-01') & (month_ftse.index < '2024-01-01')]['FTSE Open Price']
train_x = train_x.loc[train_y.index]
train_y_high=month_ftse[month_ftse.index < '2023-01-01']['FTSE High']
test_y_high = month_ftse[(month_ftse.index >= '2023-01-01') & (month_ftse.index < '2024-01-01')]['FTSE High']
train_y_low=month_ftse[month_ftse.index < '2023-01-01']['FTSE Low']
test_y_low = month_ftse[(month_ftse.index >= '2023-01-01') & (month_ftse.index < '2024-01-01')]['FTSE Low']

def backward_selection_extended(x, y, sl=0.05):
    numVars = len(x.columns)
    for i in range(numVars, 0, -1):
        regressor_OLS = sm.OLS(y, sm.add_constant(x)).fit()
        maxVar = max(regressor_OLS.pvalues[1:])
        if maxVar > sl:
            for j in range(1, i + 1):
                if regressor_OLS.pvalues[j] == maxVar:
                    x = x.drop(x.columns[j - 1], axis=1)
                    break
    return x

data_backward = backward_selection_extended(train_x, train_y, 0.05)
model_backward = sm.OLS(train_y, sm.add_constant(data_backward)).fit()
print(model_backward.summary())

data_backward_high = backward_selection_extended(train_x, train_y_high, 0.05)
model_backward_high = sm.OLS(train_y_high, sm.add_constant(data_backward_high)).fit()
print(model_backward_high.summary())

data_backward_low = backward_selection_extended(train_x, train_y_low, 0.05)
model_backward_low = sm.OLS(train_y_low, sm.add_constant(data_backward_low)).fit()
print(model_backward_low.summary())

print('Coefficients:', model_backward.params)
print('Coefficients_high:', model_backward_high.params)
print('Coefficients_low:', model_backward_low.params)

x_test_best_backward = sm.add_constant(test_x[data_backward.columns])
f_backward_2023 = model_backward.predict(x_test_best_backward)

x_test_best_backward_high = sm.add_constant(test_x[data_backward_high.columns])
f_backward_2023_high = model_backward_high.predict(x_test_best_backward_high)

x_test_best_backward_low = sm.add_constant(test_x[data_backward_low.columns])
f_backward_2023_low = model_backward_low.predict(x_test_best_backward_low)

ftse_high = month_ftse[(month_ftse.index >= '2023-01-01') & (month_ftse.index < '2024-01-01')]['FTSE High']
ftse_low = month_ftse[(month_ftse.index >= '2023-01-01') & (month_ftse.index < '2024-01-01')]['FTSE Low']

plt.figure(figsize=(10, 6))
plt.plot(test_y.index, test_y, label='Actual')
plt.plot(f_backward_2023.index, f_backward_2023, label='Forecast_backward')
plt.plot(f_backward_2023.index, f_backward_2023_high, label='Forecast_backward High')
plt.plot(f_backward_2023.index, f_backward_2023_low, label='Forecast_backward Low')
plt.fill_between(f_backward_2023.index, ftse_high, ftse_low, color='pink', alpha=0.3, label='Actual Range')
plt.legend()
plt.title('Forecast 2023 Backward')
plt.show()

########################
# 2023_residuals
########################
residuals=test_y-f_backward_2023
plot_acf(residuals,lags=11)
plt.title('ACF of Residuals_Backward')
plt.show()

probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_Backward')
plt.show()

#########
# Model 
#########
model_backward_ols = sm.OLS(train_y, sm.add_constant(data_backward)).fit()
print("Backward Selection Model Evaluation:")
print(f"R-squared: {model_backward_ols.rsquared}")
print(f"Adjusted R-squared: {model_backward_ols.rsquared_adj}")
print(f"AIC: {model_backward_ols.aic}")
print(f"BIC: {model_backward_ols.bic}")

print("\nBackward Selection Model Summary:")
print(model_backward_ols.summary())

##############################################################################
# Multiple Regression (forward)
##############################################################################
def forward_selection(X, y, sl=0.05):
    initial_features = X.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < sl:
            best_features.append(new_pval.idxmin())
        else:
            break
    return X[best_features]

data_forward = forward_selection(train_x, train_y, 0.05)
model_forward = sm.OLS(train_y, sm.add_constant(data_forward)).fit()
print(model_forward.summary())

data_forward_high = forward_selection(train_x, train_y_high, 0.05)
model_forward_high = sm.OLS(train_y_high, sm.add_constant(data_forward_high)).fit()
print(model_forward_high.summary())

data_forward_low = forward_selection(train_x, train_y_low, 0.05)
model_forward_low = sm.OLS(train_y_low, sm.add_constant(data_forward_low)).fit()
print(model_forward_low.summary())

print('Coefficients:', model_forward.params)
print('Coefficients_high:', model_forward_high.params)
print('Coefficients_low:', model_forward_low.params)

x_test_best_forward = sm.add_constant(test_x[data_forward.columns])
f_forward_2023 = model_forward.predict(x_test_best_forward)

x_test_best_forward_high = sm.add_constant(test_x[data_forward_high.columns])
f_forward_2023_high = model_forward_high.predict(x_test_best_forward_high)

x_test_best_forward_low = sm.add_constant(test_x[data_forward_low.columns])
f_forward_2023_low = model_forward_low.predict(x_test_best_forward_low)

plt.figure(figsize=(10, 6))
plt.plot(test_y.index, test_y, label='Actual')
plt.plot(f_forward_2023.index, f_forward_2023, label='Forecast_forward')
plt.plot(f_forward_2023.index, f_forward_2023_high, label='Forecast_forward High')
plt.plot(f_forward_2023.index, f_forward_2023_low, label='Forecast_forward Low')
plt.fill_between(f_forward_2023.index, ftse_high, ftse_low, color='pink', alpha=0.3, label='Actual Range')
plt.legend()
plt.title('Forecast 2023 Forward')
plt.show()

########################
# 2023_residuals
########################
residuals_forward=test_y-f_forward_2023
plot_acf(residuals_forward,lags=11)
plt.title('ACF of Residuals_Forward')
plt.show()

probplot(residuals_forward, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_Forward')
plt.show()

#########
# Model 
#########
model_forward_ols = sm.OLS(train_y, sm.add_constant(data_forward)).fit()
print("Forward Selection Model Evaluation:")
print(f"R-squared: {model_forward_ols.rsquared}")
print(f"Adjusted R-squared: {model_forward_ols.rsquared_adj}")
print(f"AIC: {model_forward_ols.aic}")
print(f"BIC: {model_forward_ols.bic}")

print("\nForward Selection Model Summary:")
print(model_forward_ols.summary())

##############################################################################
# Multiple Regression (stepwise)
##############################################################################
def stepwise_selection(X, y, sl_enter=0.05, sl_exit=0.05):
    initial_features = X.columns.tolist()
    best_features = []
    
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < sl_enter:
            best_features.append(new_pval.idxmin())
        else:
            break
    
    while len(best_features) > 0:
        model = sm.OLS(y, sm.add_constant(X[best_features])).fit()
        max_p_value = model.pvalues[1:].max()  
        if max_p_value > sl_exit:
            excluded_feature = model.pvalues[1:].idxmax()
            best_features.remove(excluded_feature)
        else:
            break 
    
    return X[best_features]

data_stepwise = stepwise_selection(train_x, train_y, 0.05)
model_stepwise = sm.OLS(train_y, sm.add_constant(data_stepwise)).fit()
print(model_stepwise.summary())

data_stepwise_high = stepwise_selection(train_x, train_y_high, 0.05)
model_stepwise_high = sm.OLS(train_y_high, sm.add_constant(data_forward_high)).fit()
print(model_stepwise_high.summary())

data_stepwise_low = stepwise_selection(train_x, train_y_low, 0.05)
model_stepwise_low = sm.OLS(train_y_low, sm.add_constant(data_stepwise_low)).fit()
print(model_stepwise_low.summary())

print('Coefficients:', model_stepwise.params)
print('Coefficients_high:', model_stepwise_high.params)
print('Coefficients_low:', model_stepwise_low.params)

x_test_best_stepwise = sm.add_constant(test_x[data_stepwise.columns])
f_stepwise_2023 = model_stepwise.predict(x_test_best_stepwise)

x_test_best_stepwise_high = sm.add_constant(test_x[data_stepwise_high.columns])
f_stepwise_2023_high = model_stepwise_high.predict(x_test_best_stepwise_high)

x_test_best_stepwise_low = sm.add_constant(test_x[data_stepwise_low.columns])
f_stepwise_2023_low = model_stepwise_low.predict(x_test_best_stepwise_low)

plt.figure(figsize=(10, 6))
plt.plot(test_y.index, test_y, label='Actual')
plt.plot(f_stepwise_2023.index, f_stepwise_2023, label='Forecast_stepwise')
plt.plot(f_stepwise_2023.index, f_stepwise_2023_high, label='Forecast_stepwise High')
plt.plot(f_stepwise_2023.index, f_stepwise_2023_low, label='Forecast_stepwise Low')
plt.fill_between(f_stepwise_2023.index, ftse_high, ftse_low, color='pink', alpha=0.3, label='Actual Range')
plt.legend()
plt.title('Forecast 2023 Stepwise')
plt.show()

########################
# 2023_residuals
########################
residuals_stepwise=test_y-f_stepwise_2023
plot_acf(residuals_stepwise,lags=11)
plt.title('ACF of Residuals_Stepwise')
plt.show()

probplot(residuals_stepwise, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals_Stepwise')
plt.show()

#########
# Model 
#########
model_stepwise_ols = sm.OLS(train_y, sm.add_constant(data_stepwise)).fit()
print("Stepwise Selection Model Evaluation:")
print(f"R-squared: {model_stepwise_ols.rsquared}")
print(f"Adjusted R-squared: {model_stepwise_ols.rsquared_adj}")
print(f"AIC: {model_stepwise_ols.aic}")
print(f"BIC: {model_stepwise_ols.bic}")

print("\nForward Selection Model Summary:")
print(model_stepwise_ols.summary())

########################
# Forecasting Backward_2024 
########################
f_x = data.loc[(data.index >= '2024-01-01') & (data.index < '2025-01-01'), extended_x]
x_f_best_backward = sm.add_constant(f_x[data_backward.columns])
f_f_2023 = model_backward.predict(x_f_best_backward)
print(f_f_2023)
x_f_best_backward_high = sm.add_constant(f_x[data_backward_high.columns])
f_f_2023_high = model_backward_high.predict(x_f_best_backward_high)
print(f_f_2023_high)
x_f_best_backward_low = sm.add_constant(f_x[data_backward_low.columns])
f_f_2023_low = model_backward_low.predict(x_f_best_backward_low)
print(f_f_2023_low)

x_f_best_forward = sm.add_constant(f_x[data_forward.columns])
f_f_2023 = model_forward.predict(x_f_best_forward)
print(f_f_2023)
x_f_best_forward_high = sm.add_constant(f_x[data_forward_high.columns])
f_f_2023_high = model_forward_high.predict(x_f_best_forward_high)
print(f_f_2023_high)
x_f_best_forward_low = sm.add_constant(f_x[data_forward_low.columns])
f_f_2023_low = model_forward_low.predict(x_f_best_forward_low)
print(f_f_2023_low)

x_f_best_stepwise = sm.add_constant(f_x[data_stepwise.columns])
f_f_2023 = model_stepwise.predict(x_f_best_stepwise)
print(f_f_2023)
x_f_best_stepwise_high = sm.add_constant(f_x[data_stepwise_high.columns])
f_f_2023_high = model_stepwise_high.predict(x_f_best_stepwise_high)
print(f_f_2023_high)
x_f_best_stepwise_low = sm.add_constant(f_x[data_stepwise_low.columns])
f_f_2023_low = model_stepwise_low.predict(x_f_best_stepwise_low)
print(f_f_2023_low)
