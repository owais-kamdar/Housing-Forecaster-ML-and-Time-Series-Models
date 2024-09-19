from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from visualize import plot_real_vs_predicted
from visualize import plot_model_comparisons

def train_test_split_data(region_data, features, target):
    return train_test_split(region_data[features], region_data[target], test_size=0.2, random_state=42)

# Function to scale features and tune Ridge Regression
def tune_ridge_regression(X_train, y_train):
    # Create a pipeline to scale data and fit Ridge Regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale the data
        ('ridge', Ridge())  # Ridge regression model
    ])

    # Define hyperparameter grid for tuning
    params = {
        'ridge__alpha': [0.01, 0.1, 1.0, 10, 100],  # Tune alpha
        'ridge__fit_intercept': [True, False],  # Test with/without intercept
        'ridge__normalize': [True, False]  # Test normalization
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and hyperparameters
    best_ridge_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best parameters for Ridge Regression: {best_params}")
    
    return best_ridge_model

def train_and_predict_model(model_choice, X_train, X_test, y_train, y_test):
    if model_choice == '1':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = 'Random Forest'
    elif model_choice == '2':
        model = Ridge(alpha=1.0)
        model_name = 'Ridge Regression'
    elif model_choice == '3':
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression()
        model_name = 'Polynomial Regression'
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        print("Invalid selection. Please select a valid model.")
        return
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model_name, y_pred

# Function to evaluate the performance of Ridge Regression
def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MAE: {mae}, RMSE: {rmse}, R2: {r2}")

    return mae, rmse, r2

