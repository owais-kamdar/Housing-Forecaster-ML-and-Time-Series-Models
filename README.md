# Housing Forecaster

This project utilizes time series analysis and machine learning models to predict housing prices (ZHVI) and rent prices (ZORI) for various regions. The project implements SARIMA and ARIMA models for time series forecasting, alongside machine learning models such as Random Forest, Ridge Regression, and Polynomial Regression for predictions.

## Project Structure
- **Data:** Multiple CSV files containing housing market and real estate data.
- **Models:** SARIMA, ARIMA, Random Forest, Ridge Regression, Polynomial Regression.
- **Evaluation:** Comparison of model performances using metrics like MAE and RMSE, and visualization of actual vs predicted values.

## Project Structure
- **data/:** Directory containing multiple CSV files with housing market and real estate data.
- **load_data.py:** Script responsible for loading, reshaping, and combining datasets.
- **model.py:** Contains machine learning models for prediction and their evaluation.
- **timeseries_model.py:** Implements time series models (SARIMA and ARIMA) for forecasting.
- **visualize.py:** Contains functions for visualizing model results, including real vs predicted - values and model comparisons.
- **main.py:** Main script to execute machine learning model training and predictions.
- **timeseries_main.py:** Main script to execute time series model predictions.
- **requirements.txt:** File containing the list of dependencies for the project.


## Setup and Installation

### Requirements
- Python 3.8+
- Install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Setup
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/owais-kamdar/housing-forecaster.git
   ```

2. Navigate to the project directory:

   ```bash
   cd housing-forecaster
   ```

3. Install the required libraries (if not already installed):

   ```bash
   pip install -r requirements.txt
   ```

4. Run the machine learning models by executing the main.py script:

   ```bash
   python main.py


   ```

5. Run the time series models by executing the timeseries_main.py script:

   ```bash
   python timeseries_main.py
   ```
   

## Results
The project compares various models for predicting housing and rent prices. Graphical comparisons of real vs predicted values are provided for model evaluation.

Example Visualizations:
Real vs Predicted values for different models.
Model performance comparisons (e.g., MAE, RMSE).


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
