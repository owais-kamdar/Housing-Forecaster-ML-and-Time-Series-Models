import sys
from load_data import load_datasets, combine_datasets
from model import train_and_predict_model, train_test_split_data, evaluate_model
from visualize import plot_real_vs_predicted, plot_model_comparisons

def main():
    loaded_data = load_datasets()
    combined_data = combine_datasets(loaded_data)

    region = input("Enter the region (e.g., 'New York, NY'): ")

    # Select house or rent prices
    print("Select what you want to predict:")
    print("(1) House Prices (ZHVI)")
    print("(2) Rent Prices (ZORI)")
    target_choice = input("Enter your choice (1 or 2): ")
    target = 'zhvi_value' if target_choice == '1' else 'zori_value'

    features = ['for_sale_value', 'price_cuts_value', 'new_listings_value']

    # Filter data
    region_data = combined_data[combined_data['RegionName'] == region].dropna(subset=features + [target])
    if region_data.empty:
        print(f"No data available for the region '{region}'.")
        sys.exit()

    # After filtering the region-specific data
    print(f"Region-specific data has {region_data.shape[0]} rows")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(region_data, features, target)
    model_results = {}  # Dictionary to store predictions from all models
    model_colors = {'Random Forest': 'red', 'Ridge Regression': 'green', 'Polynomial Regression': 'orange'}  # Colors for each model

    while True:
        print("\nSelect a model:")
        print("(1) Random Forest")
        print("(2) Ridge Regression")
        print("(3) Polynomial Regression")
        print("(4) Quit and plot all models")
        model_choice = input("Enter your choice: ")
        if model_choice == '4':
            break

        model_name, y_pred = train_and_predict_model(model_choice, X_train, X_test, y_train, y_test)
        if model_name:
            evaluate_model(y_test, y_pred, model_name)

            # Store the predictions
            model_results[model_name] = y_pred

            # Print model results to check if multiple models are stored
            print(f"Stored results for model: {model_name}")
            print(f"Current model results: {model_results.keys()}")

            # Assuming region_data contains the Date column
            dates = region_data.loc[X_test.index, 'Date'].sort_values().dt.strftime('%b-%Y').reset_index(drop=True)

            # Call the function with the dates to plot individual model predictions
            plot_real_vs_predicted(y_test, y_pred, model_name, dates)

    # After training all models, overlay their predictions on a single graph
    if model_results:
        plot_model_comparisons(y_test, model_results, dates, model_colors)

if __name__ == "__main__":
    main()
