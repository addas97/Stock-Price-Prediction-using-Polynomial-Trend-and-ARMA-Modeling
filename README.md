# Stock-Price-Prediction-using-Polynomial-Trend-and-ARMA-Modeling
Forecasts stock prices using polynomial regression for trend and ARMA for residuals. Pulls data from Yahoo Finance, selects optimal model via AIC/BIC, and generates future predictions with confidence intervals. Combines trend and short-term fluctuations for robust forecasting.

📈 Stock Price Forecasting using Polynomial Trend and ARMA Residual Modeling
This project presents a robust stock price forecasting framework that models both long-term trends and short-term fluctuations in stock prices. It combines polynomial regression to capture overall movement and ARMA (AutoRegressive Moving Average) modeling to explain residual variance. Forecasts include confidence intervals for uncertainty quantification.

🔍 Project Overview
Using historical closing prices pulled from Yahoo Finance, the model forecasts future prices through a two-step pipeline:

🧹 Data Loading & Transformation
User inputs a ticker (e.g., AAPL, META, etc.).
Data is fetched via yfinance from 2008 to present.
Log transformation applied to prices to stabilize variance and linearize growth.

📐 Trend Modeling with Polynomial Regression
Fits polynomial models (degree 1 to 3) to the log-transformed price.
Selects the best model based on RMSE.
Saves or reloads the trained model with joblib for efficiency.

🔁 Residual Modeling with ARMA
Extracts residuals from the trend model and evaluates their autocorrelation structure via ACF/PACF.
Performs grid search over ARMA(p, q) orders (0 ≤ p, q ≤ 10) to minimize AIC and BIC.
Trains ARMA model on residuals and validates with:
Ljung-Box Test: Checks if residuals are white noise.
Residual Plots and ACF: Ensures adequacy of model fit.

🔮 Forecasting
Combines polynomial trend forecasts with ARMA-predicted residuals.
Forecasts into the future (e.g., 30 days).

📊 Plots:
Original prices
Predicted prices
95% confidence intervals

📊 Model Evaluation
RMSE calculated on both training and test sets.
Final forecasts plotted against recent stock history.
All models (polynomial and ARMA) are saved and reused for consistency and speed.

⚙️ Technologies & Tools
Python (NumPy, Pandas, Matplotlib, Scikit-learn)
yfinance for stock data
Statsmodels (ARIMA, Ljung-Box Test, ACF/PACF)
joblib for model caching
