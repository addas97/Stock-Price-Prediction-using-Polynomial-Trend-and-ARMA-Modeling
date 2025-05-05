# Akash Das
# Stock Price Forecasting 

import os 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib

import warnings
warnings.filterwarnings("ignore")

# == Load Data ==
def ticker():
    return input('Enter a ticker: ').upper()

def get_data(ticker):
    stock = yf.download(ticker, start = '2008-01-01', period='1d')
    data = pd.DataFrame(stock)
    prices = np.array(data['Close']).reshape(-1, 1)
    data.index = pd.to_datetime(data.index)
    dates = data.index.to_numpy().reshape(-1, 1)
    index = np.arange(len(prices)).reshape(-1, 1)

    return (prices, dates, index) if len(prices) == len(dates) == len(index) else None

ticker = ticker()
prices, dates, index = get_data(ticker)

# Plot Function
def plot(dates, prices, ticker, title):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices)
    plt.title(f'{ticker} {title}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()
    
# == Transform Data ==
def transform_px(prices):
    return np.log(prices) # Log prices → smoother, more stable for modeling

log_prices = transform_px(prices).flatten()

plot(dates, prices, ticker, title='Prices Over Time')
plot(dates, log_prices, ticker, title='Log-Transformed Prices Over Time')

# == Fit Trend Models ==

# Split Data into Training / Testing Set
training_set_len = int(len(prices) * .80)
training_set_price = log_prices[:training_set_len]
training_set_index = index[:training_set_len]
testing_set_price = log_prices[training_set_len:]
testing_set_index = index[training_set_len:]

# Fit model on best degree 
def rmse(true_px, preds):
    return np.mean((true_px - preds) ** 2) ** 0.5

RMSE_linear = []

def fit_polynomial(degree, index, prices):
    model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
    model.fit(index, prices)
    poly_preds = model.predict(index)
    poly_residuals = prices - poly_preds
    RMSE_linear.append(rmse(prices, poly_preds))
    return model, poly_preds, poly_residuals

if not os.path.exists(f'poly_model_{ticker}.pkl'):
    
    # Select Best Degree Fit
    for i in range(1, 4):
        fit_polynomial(degree = i, index = training_set_index, prices = training_set_price)

    degree = np.argmin(RMSE_linear) + 1
    poly_model, preds_training_poly, residuals_training_poly = fit_polynomial(degree = degree, index = training_set_index, prices = training_set_price)

    joblib.dump({'model' : poly_model,
                 'preds' : preds_training_poly, 
                 'residuals' : residuals_training_poly, 
                 'deg' : degree}, 
                 f'poly_model_{ticker}.pkl')
    
    print(f'Saved polynomial model for {ticker} with degree {degree}...')

else:
    data = joblib.load(f'poly_model_{ticker}.pkl')
    print(f'Loaded saved polynomial model for {ticker}...')
    poly_model = data['model']
    preds_training_poly = data['preds']
    residuals_training_poly = data['residuals']
    degree = data['deg']

# Plot fit
'''plt.figure(figsize=(10, 5))
plt.plot(dates, log_prices)
plt.plot(dates[:training_set_len], preds_training_poly)
plt.title(f'{ticker} {degree} Degree Fit')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()
plot(dates, residuals_training_poly, ticker, title = f'{degree}th Degree Fit Residuals')'''

# == Fit ARMA Models ==
# Determine model orders via ACF / PACF

# ACF - slow decay indicates non-stationary / strongly correlated time series
sm.graphics.tsa.plot_acf(residuals_training_poly, lags=30)
plt.show()

#PACF
sm.graphics.tsa.plot_pacf(residuals_training_poly, lags=30)
plt.show()

'''
The slow decay of ACF implies a non-stationary / strongly correlated time series. 
We will proceed with an ARMA model and then evaluate performance.
Ex: For META, we see a large drop off on the PACF after order = 1.
'''

# Training for ARMA Model
def grid_search_params(residuals, AR_range, MA_range):
    min_aic = np.inf
    min_bic = np.inf
    min_aic_index = None
    min_bic_index = None
    aic_matrix = np.zeros((len(AR_range), len(MA_range)))
    bic_matrix = np.zeros((len(AR_range), len(MA_range)))

    for AR_order in AR_range:
        for MA_order in MA_range:
            arma = ARIMA(residuals, order = (AR_order, 0, MA_order)).fit()
            aic_matrix[AR_order, MA_order] = arma.aic
            bic_matrix[AR_order, MA_order] = arma.bic

            if arma.aic < min_aic:
                min_aic = arma.aic
                min_aic_index = (AR_order, 0, MA_order)

            if arma.bic < min_bic:
                min_bic = arma.bic
                min_bic_index = (AR_order, 0, MA_order)

        print(f"Training current AR order: {AR_order}")

    return min_aic_index, aic_matrix, min_bic_index, bic_matrix

if os.path.exists(f'best_orders_{ticker}.pkl'):
    orders = joblib.load(f'best_orders_{ticker}.pkl')
    min_aic_index = orders['min_aic_index']
    min_bic_index = orders['min_bic_index']
    print(f'Loaded saved ARMA model orders for {ticker}...')

else:
    min_aic_index, _, min_bic_index, _ = grid_search_params(residuals_training_poly, range(11), range(11))
    joblib.dump({'min_aic_index': min_aic_index, 'min_bic_index': min_bic_index}, f'best_orders_{ticker}.pkl')
    print(f"Saved best orders for {ticker}: AIC {min_aic_index}, BIC {min_bic_index}")

def model_performance(min_aic_index, min_bic_index):
    arma_aic = ARIMA(residuals_training_poly, order = min_aic_index).fit()
    rmse_arma_aic = np.mean(arma_aic.resid ** 2) ** 0.5

    arma_bic = ARIMA(residuals_training_poly, order = min_bic_index).fit()
    rmse_arma_bic = np.mean(arma_bic.resid ** 2) ** 0.5

    if rmse_arma_aic < rmse_arma_bic:
        print("Automatic selection finds model with AR {0}, MA {2}".format(*min_aic_index))
        print("RMSE with selected model (AIC): ", rmse_arma_aic)
        print(arma_aic.summary())
        joblib.dump({'model': arma_aic, 
                     'order': min_aic_index}, f'arma_{ticker}.pkl')
        return arma_aic, min_aic_index
    
    else:
        print("Automatic selection finds model with AR {0}, MA {2}".format(*min_bic_index))
        print("RMSE with selected model (BIC): ", rmse_arma_bic)
        print(arma_bic.summary())
        joblib.dump({'model': arma_bic, 
                     'order': min_bic_index}, f'arma_{ticker}.pkl')
        return arma_bic, min_bic_index

if os.path.exists(f'arma_{ticker}.pkl') == False:
    arma_model, order = model_performance(min_aic_index, min_bic_index)
    arma_preds = arma_model.predict()
    arma_residuals = arma_model.resid

    # Evaluate Model: ARMA Model Performance & Plotting
    # Ljung-Box -- p_value > 0.05, residuals are white noise --> residuals do not have meaningful autocorrelation beyond chosen lag
    # Ljung-Box test checks whether any autocorrelation remains in the residuals up to a certain lag. Ljung-Box test is always tied to the residuals of a specific fitted model.
    ljungbox = []
    for lag in range(10, 100, 10):
        ljungbox.append(acorr_ljungbox(arma_residuals, lags = [lag])['lb_pvalue'])

    plt.figure(figsize=(10, 5))
    plt.title('Ljung-Box for Various Lags')
    plt.axvline(np.sum(np.array(order)), color = 'r', linestyle = '--', label = 'Model Lag (AR + MA)')
    plt.xlabel('Lags')
    plt.ylabel('Ljung-Box p-value')
    plt.plot(range(10, 100, 10), ljungbox)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Residual Comparisons between Trend Fit vs. ARMA Model Fit')
    plt.plot(residuals_training_poly, label = 'Residuals from fitted Polynomial')
    plt.plot(arma_residuals, label = 'Fitted with ARMA process')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('ARMA Risduals')
    plt.plot(arma_residuals, marker = 'o')
    plt.show()

    sm.graphics.tsa.plot_acf(arma_residuals, lags = 50)
    plt.title('Residual ACF After ARIMA')
    plt.show()

else:
    data_arma = joblib.load(f'arma_{ticker}.pkl')
    arma_model = data_arma['model']
    order = data_arma['order']
    print(f'Loaded saved ARMA model for {ticker}...')
    arma_preds = arma_model.predict()
    arma_residuals = arma_model.resid

final_model_price_training = preds_training_poly + residuals_training_poly
print(f'RMSE on training set on full Poly and ARMA model: {rmse(training_set_price, final_model_price_training)}')

# Evaluate Test Set
poly_preds = poly_model.predict(testing_set_index).flatten()
arma_test_residuals = arma_model.forecast(steps = len(testing_set_index)).flatten()
tot_model_pred_price = poly_preds + arma_test_residuals
rmse_final_pred = rmse(testing_set_price, tot_model_pred_price)
print(f"Test Set RMSE: {rmse_final_pred}")

# == Retrain Model on Entire Series ==

# Polynomial Model Fit
full_model_poly = poly_model.fit(index, log_prices)
full_model_poly_preds = full_model_poly.predict(index).flatten()
full_model_poly_resid = log_prices - full_model_poly_preds.flatten()

# ARMA Model Fit
full_model_arma = ARIMA(endog = full_model_poly_resid, order = order).fit()
full_model_arma_residuals = full_model_arma.resid

# Forecast Into Future
h = 30 # steps into the future

# Forecast Trend
future_index = np.arange(len(index), len(index) + h).reshape(-1, 1)
future_trend_preds = full_model_poly.predict(future_index)

# Forecast Residuals
forecast_result = full_model_arma.get_forecast(steps = h)
future_resid_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Combine
future_log_mean = future_trend_preds + future_resid_mean
future_log_lower = future_trend_preds + conf_int[:, 0]
future_log_upper = future_trend_preds + conf_int[:, 1]

plt.figure(figsize=(10, 5))
plt.plot(range(len(index) - 200, len(index)), np.exp(log_prices[len(index) - 200:].flatten()), label = 'Original Data')

# Forecasted price mean
plt.plot(future_index.flatten(), np.exp(future_log_mean), color='r', linestyle='--', label=f'Predicted Price ({h} Steps)')

# Confidence interval
plt.fill_between(future_index.flatten(), np.exp(future_log_lower), np.exp(future_log_upper), color='r', alpha=0.3, label='95% Confidence Interval')

print(f'Low-High Price Range for Mean Predicted Price: {round(float(np.exp(min(future_log_mean))), 2), round(float(np.exp(max(future_log_mean))), 2)}')

plt.title(f"{ticker} Stock Prices Over Time with {h}-Step Predictions")
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.axvline(x= len(index) - 1, color='b', linestyle='--', label='Forecast Start')
plt.legend()
plt.tight_layout()
plt.show()