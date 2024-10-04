# load_data.py
import os
import pandas as pd

def load_datasets():
    data_folder = 'data'  # Folder where the CSV files are stored
    datasets = {
        'zhvi': 'zhvi.csv',
        'zori': 'zori.csv',
        'for_sale': 'for_sale.csv',
        'new_listings': 'new_listings.csv',
        'price_cuts': 'median-price-cuts.csv'
    }
    return {name: pd.read_csv(os.path.join(data_folder, path)) for name, path in datasets.items()}

def reshape_and_filter(df, date_start='2018-01-01', date_format='%Y-%m-%d'):
    df_long = pd.melt(df, id_vars='RegionName', var_name='Date', value_name='Value')
    df_long['Date'] = pd.to_datetime(df_long['Date'], format=date_format, errors='coerce')
    return df_long[df_long['Date'] >= pd.to_datetime(date_start)]

def combine_datasets(loaded_data):
    combined_df_list = []
    for name, df in loaded_data.items():
        reshaped_df = reshape_and_filter(df)
        reshaped_df.rename(columns={'Value': f'{name}_value'}, inplace=True)
        combined_df_list.append(reshaped_df)
    combined_data = combined_df_list[0]
    for df in combined_df_list[1:]:
        combined_data = pd.merge(combined_data, df, on=['RegionName', 'Date'], how='outer')
    return combined_data
