from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from load_data import load_datasets, combine_datasets

# Function to evaluate the model performance
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")

# Function to plot actual vs all model predictions
def plot_model_comparisons(y_test, predictions_dict):
    plt.figure(figsize=(10, 6))
    
    # Plot actual values
    plt.plot(range(len(y_test)), y_test, label='Actual Values', color='blue')

    # Plot each model's predictions
    for model_name, y_pred in predictions_dict.items():
        plt.plot(range(len(y_pred)), y_pred, label=f'{model_name} Predictions')

    plt.title("Comparison of Model Predictions")
    plt.xlabel("Data Points")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Load and combine datasets
loaded_data = load_datasets()
combined_data = combine_datasets(loaded_data)

# Filter the dataset for a specific region (e.g., New York)
region_data = combined_data[combined_data['RegionName'].str.contains('New York', na=False)].dropna()

# Features and target column for training the models
features = ['for_sale_value', 'price_cuts_value', 'new_listings_value']
target = 'zhvi_value'

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(region_data[features], region_data[target], test_size=0.2, random_state=42)

# Dictionary to store predictions from different models
predictions_dict = {}

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
evaluate_model(y_test, rf_pred, "Random Forest")
predictions_dict['Random Forest'] = rf_pred

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
evaluate_model(y_test, ridge_pred, "Ridge Regression")
predictions_dict['Ridge Regression'] = ridge_pred

# Train Polynomial Regression model
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)
evaluate_model(y_test, poly_pred, "Polynomial Regression")
predictions_dict['Polynomial Regression'] = poly_pred

# Plot all model comparisons on the same graph
plot_model_comparisons(y_test, predictions_dict)
