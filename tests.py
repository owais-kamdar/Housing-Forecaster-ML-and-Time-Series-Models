# tests.py

import unittest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from model import evaluate_model
from load_data import load_datasets, combine_datasets
from visualize import plot_model_comparisons

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded_data = load_datasets()
        combined_data = combine_datasets(loaded_data)
        cls.region_data = combined_data[combined_data['RegionName'] == 'New York, NY'].dropna()
        cls.features = ['for_sale_value', 'price_cuts_value', 'new_listings_value']
        cls.target = 'zhvi_value'
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.region_data[cls.features], cls.region_data[cls.target], test_size=0.2, random_state=42
        )

    def test_random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        evaluate_model(self.y_test, y_pred, "Random Forest")
        self.assertIsNotNone(y_pred)

    def test_ridge_regression(self):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        evaluate_model(self.y_test, y_pred, "Ridge Regression")
        self.assertIsNotNone(y_pred)

    def test_polynomial_regression(self):
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('linear', LinearRegression())
        ])
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        evaluate_model(self.y_test, y_pred, "Polynomial Regression")
        self.assertIsNotNone(y_pred)

if __name__ == '__main__':
    unittest.main()
