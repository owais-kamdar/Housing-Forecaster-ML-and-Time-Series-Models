import pandas as pd
from load_data import load_datasets, combine_datasets
from timeseries_model import sarima_model, arima_model, future_forecast, prepare_data, plot_predictions, evaluate_predictions

def run_forecasting_comparison(data, region, target_column):
    """
    Handle the comparison between SARIMA and ARIMA models for forecasting.
    """
    # Prepare the data for modeling
    region_data = prepare_data(data, region, target_column)
    if region_data is None:
        return

    # Get forecast horizon in years and convert to months
    forecast_years = int(input("Enter the number of years into the future for forecasting: "))
    steps = forecast_years * 12

    # Fit SARIMA model
    print("\nFitting SARIMA model...")
    sarima_fit = sarima_model(region_data)
    future_predictions_sarima = future_forecast(sarima_fit, steps=steps)

    # Fit ARIMA model
    print("\nFitting ARIMA model...")
    arima_fit = arima_model(region_data)
    future_predictions_arima = future_forecast(arima_fit, steps=steps)

    # Plot SARIMA and ARIMA predictions
    plot_predictions(region_data, future_predictions_sarima, future_predictions_arima, region)

    # Evaluate predictions from both models
    evaluate_predictions(region_data[-steps:], future_predictions_sarima, future_predictions_arima)

def main():
    # Load datasets and combine them
    loaded_data = load_datasets()
    combined_data = combine_datasets(loaded_data)

    # Get user inputs for region and target column
    region = input("Enter the region (e.g., 'New York, NY'): ")
    print("Select what you want to predict:")
    print("(1) House Prices (ZHVI)")
    print("(2) Rent Prices (ZORI)")
    target_choice = input("Enter your choice (1 or 2): ")
    target = 'zhvi_value' if target_choice == '1' else 'zori_value'

    # Run the forecasting comparison between SARIMA and ARIMA
    run_forecasting_comparison(combined_data, region, target)

if __name__ == "__main__":
    main()
