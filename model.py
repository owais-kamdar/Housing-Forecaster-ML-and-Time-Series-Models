from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

def train_test_split_data(region_data, features, target):
    # Check how many data points are being used
    print(f"Total number of data points: {region_data.shape[0]}")

    # Ensure the data isn't accidentally filtered down
    return train_test_split(region_data[features], region_data[target], test_size=0.2, random_state=42)

def train_and_predict_model(model_choice, X_train, X_test, y_train, y_test):
    if model_choice == '1':
        model_name = 'Random Forest'
        pipeline = Pipeline([('model', RandomForestRegressor(random_state=42))])
    elif model_choice == '2':
        model_name = 'Ridge Regression'
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
    elif model_choice == '3':
        model_name = 'Polynomial Regression'
        pipeline = Pipeline([('poly_features', PolynomialFeatures(degree=2)), ('scaler', StandardScaler()), ('model', LinearRegression())])
    else:
        print("Invalid selection. Please select a valid model.")
        return None, None

    # Fit the model
    model = pipeline.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model_name, y_pred

def evaluate_model(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return mae, rmse, r2
