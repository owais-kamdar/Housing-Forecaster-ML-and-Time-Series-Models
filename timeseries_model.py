import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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
    print(f"SARIMA - MAE: {mae_sarima:.2f}, MSE: {mse_sarima:.2f}")
    print(f"ARIMA - MAE: {mae_arima:.2f}, MSE: {mse_arima:.2f}")
    return mse_sarima, mae_sarima, mse_arima, mae_arima

def prepare_data(data, region, target_column):
    """
    Prepares the data for time series modeling by filtering the region
    and selecting the target column.
    """
    region_data = data[data['RegionName'] == region]
    if region_data.empty:
        print(f"No data available for the region '{region}'.")
        return None

    # Convert the date column to datetime and set it as the index
    region_data['Date'] = pd.to_datetime(region_data['Date'])
    region_data.set_index('Date', inplace=True)
    
    # Ensure the target column has no missing values
    if region_data[target_column].isnull().sum() > 0:
        print(f"Missing values found in {target_column}. Dropping rows with missing values.")
        region_data = region_data.dropna(subset=[target_column])
    
    # Convert the target column to numeric (float) to ensure compatibility with SARIMAX/ARIMA
    region_data[target_column] = pd.to_numeric(region_data[target_column], errors='coerce')

    # After ensuring numeric data, check for any remaining NaN values and drop them
    region_data = region_data.dropna(subset=[target_column])

    # Apply log transformation to target column to handle scaling
    region_data[target_column] = np.log1p(region_data[target_column])

    return region_data[target_column]  # Return the target column as a time series

def plot_predictions(region_data, future_predictions_sarima, future_predictions_arima, region):
    """
    Plots SARIMA and ARIMA model predictions for comparison.
    """
    plt.figure(figsize=(10, 6))  # Create a figure

    # Reverse log1p transformation for plotting the actual data
    plt.plot(region_data.index, np.expm1(region_data.values), label="Actual Data", marker='o')
    
    # Generate future dates for predictions
    future_dates = pd.date_range(start=region_data.index[-1], periods=len(future_predictions_sarima)+1, freq='M')[1:]

    # Reverse log1p transformation for predictions and plot them
    plt.plot(future_dates, np.expm1(future_predictions_sarima), label="SARIMA Predictions", color='green', linestyle='--', marker='x')
    plt.plot(future_dates, np.expm1(future_predictions_arima), label="ARIMA Predictions", color='red', linestyle='--', marker='s')
    
    plt.title(f"Comparison of SARIMA and ARIMA Predictions for {region}")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
