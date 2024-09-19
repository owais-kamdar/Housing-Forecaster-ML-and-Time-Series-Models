import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

def sarima_model(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    sarima_fit = model.fit(disp=False)
    return sarima_fit

def arima_model(train_data, order=(1, 1, 1)):
    model = ARIMA(train_data, order=order)
    arima_fit = model.fit()
    return arima_fit

def future_forecast(model_fit, steps=12):
    forecast = model_fit.get_forecast(steps=steps)
    return np.asarray(forecast.predicted_mean)

def evaluate_predictions(true_data, sarima_predictions, arima_predictions):
    mse_sarima = mean_squared_error(true_data, sarima_predictions)
    mse_arima = mean_squared_error(true_data, arima_predictions)
    mae_sarima = mean_absolute_error(true_data, sarima_predictions)
    mae_arima = mean_absolute_error(true_data, arima_predictions)

    print(f"SARIMA - MSE: {mse_sarima}, MAE: {mae_sarima}")
    print(f"ARIMA - MSE: {mse_arima}, MAE: {mae_arima}")
