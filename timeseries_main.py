from load_data import load_datasets, combine_datasets
from timeseries_model import sarima_model, arima_model, future_forecast, prepare_data, plot_predictions, evaluate_predictions
import matplotlib.pyplot as plt

# Function to handle user input for SARIMA and ARIMA comparison
def run_forecasting_comparison(data, region, target_column):
    # Prepare the data for the selected region and target column (e.g., 'zhvi_value' or 'zori_value')
    region_data_scaled, scaler = prepare_data(data, region, target_column)
    if region_data_scaled is None:
        return

    # Ask for the forecast horizon (years into the future)
    forecast_years = int(input("Enter the number of years into the future for forecasting: "))
    steps = forecast_years * 12  # Assuming monthly data

    # SARIMA Model
    sarima_fit = sarima_model(region_data_scaled)
    future_predictions_sarima_scaled = future_forecast(sarima_fit, steps=steps)
    future_predictions_sarima = scaler.inverse_transform(future_predictions_sarima_scaled.reshape(-1, 1))

    # ARIMA Model
    arima_fit = arima_model(region_data_scaled)
    future_predictions_arima_scaled = future_forecast(arima_fit, steps=steps)
    future_predictions_arima = scaler.inverse_transform(future_predictions_arima_scaled.reshape(-1, 1))

    # Inverse transform the training data for plotting
    train_data = scaler.inverse_transform(region_data_scaled)

    # Plot predictions for SARIMA and ARIMA
    plot_predictions(train_data, future_predictions_sarima, future_predictions_arima, region)

    # Evaluate both models
    evaluate_predictions(train_data[-steps:], future_predictions_sarima, future_predictions_arima)


def main():
    # Load the data and combine datasets
    loaded_data = load_datasets()
    combined_data = combine_datasets(loaded_data)

    # Ask user for region and target column
    region = input("Enter the region (e.g., 'New York, NY'): ")
    target = input("Choose a target: 'zhvi_value' for home prices or 'zori_value' for rent prices: ")

    # Run SARIMA and ARIMA comparison
    run_forecasting_comparison(combined_data, region, target)


if __name__ == "__main__":
    main()
