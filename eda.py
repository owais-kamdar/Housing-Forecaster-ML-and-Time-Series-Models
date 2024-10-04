import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_datasets, combine_datasets

def visualize_missing_data(data):
    """Visualize missing data in the dataset."""
    print("Generating heatmap for missing data...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values in the Dataset')
    plt.show()

def plot_distribution(data, column, title):
    """Plot the distribution of a specified column."""
    print(f"Generating distribution plot for {title}...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {title}')
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.show()

def plot_missing_values_by_column(data):
    """Plot the count of missing values by column."""
    print("Generating bar chart for missing values by column...")
    missing_data_count = data.isnull().sum()
    missing_data_count = missing_data_count[missing_data_count > 0]
    plt.figure(figsize=(10, 6))
    missing_data_count.plot(kind='bar', color='orange')
    plt.title('Count of Missing Values by Column')
    plt.ylabel('Missing Values Count')
    plt.show()

def correlation_heatmap(data, columns):
    """Plot a correlation heatmap for specified columns."""
    print(f"Generating correlation heatmap for {columns}...")
    corr = data[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def time_series_plot(data, region, target):
    """Plot time series data for a specific region and target variable."""
    print(f"Generating time series plot for {target} in {region}...")
    region_data = data[data['RegionName'] == region]
    if region_data.empty:
        print(f"No data available for {region}")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(region_data['Date'], region_data[target], marker='o')
    plt.title(f'Time Series of {target} in {region}')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def perform_eda(combined_data):
    # Checking missing data
    visualize_missing_data(combined_data)
    
    # Plot distribution for price_cuts_value
    plot_distribution(combined_data, 'price_cuts_value', 'Price Cuts')

    # Plot missing values by column
    plot_missing_values_by_column(combined_data)
    
    # Correlation heatmap between numeric columns
    columns_to_correlate = ['zhvi_value', 'zori_value', 'for_sale_value', 'new_listings_value', 'price_cuts_value']
    correlation_heatmap(combined_data, columns_to_correlate)
    
    # Time series plot for a specific region
    time_series_plot(combined_data, 'New York, NY', 'zhvi_value')

if __name__ == "__main__":
    # Load the datasets
    loaded_data = load_datasets()
    combined_data = combine_datasets(loaded_data)
    
    # Perform EDA on the combined dataset
    perform_eda(combined_data)
