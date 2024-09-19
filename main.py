from load_data import load_datasets, combine_datasets, preprocess_and_visualize
from model import train_and_predict_model, train_test_split_data, evaluate_model
from visualize import plot_real_vs_predicted, plot_model_comparisons

def get_user_input():
    while True:
        prediction_type = input("Select what you want to predict: (1) House Prices (ZHVI), (2) Rent Prices (ZORI): ")
        if prediction_type == '1':
            return 'zhvi_value'
        elif prediction_type == '2':
            return 'zori_value'
        else:
            print("Invalid selection. Please choose either 1 or 2.")

def get_region_input():
    region = input("Enter the region you want to analyze (e.g., 'New York, NY'): ")
    return region

def main():
    loaded_data = load_datasets()
    combined_data = combine_datasets(loaded_data)

    # preprocess_and_visualize(combined_data)

    target = get_user_input()
    region = get_region_input()
    features = ['for_sale_value', 'price_cuts_value', 'new_listings_value']

    region_data = combined_data[combined_data['RegionName'] == region].dropna(subset=features + [target])
    if region_data.empty:
        print(f"No data available for the region '{region}'")
        return

    X_train, X_test, y_train, y_test = train_test_split_data(region_data, features, target)
    model_results = {}

    while True:
        model_choice = input("Select a model: (1) Random Forest, (2) Ridge Regression, (3) Polynomial Regression, (4) Quit: ")
        if model_choice == '4':
            break
        model_name, y_pred = train_and_predict_model(model_choice, X_train, X_test, y_train, y_test)
        evaluate_model(y_test, y_pred, model_name)
        model_results[model_name] = y_pred
        plot_real_vs_predicted(y_test, y_pred, model_name)

    plot_model_comparisons(model_results, y_test)

if __name__ == "__main__":
    main()