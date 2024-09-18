import pandas as pd

# Load the datasets
zhvi_data = pd.read_csv('home_value.csv')
zri_data = pd.read_csv('rent_value.csv')
market_report_data = pd.read_csv('market_marks.csv')
new_construction_data = pd.read_csv('new_con.csv')

# Step 1: Melt the data (make sure we only melt the actual date columns)
date_columns_zhvi = zhvi_data.columns[zhvi_data.columns.str.contains(r'\d{4}-\d{2}-\d{2}')]  # Find columns with date format (YYYY-MM-DD)
date_columns_zri = zri_data.columns[zri_data.columns.str.contains(r'\d{4}-\d{2}-\d{2}')]
date_columns_market = market_report_data.columns[market_report_data.columns.str.contains(r'\d{4}-\d{2}-\d{2}')]
date_columns_construction = new_construction_data.columns[new_construction_data.columns.str.contains(r'\d{4}-\d{2}-\d{2}')]

# Melt only the date columns
zhvi_melted = zhvi_data.melt(id_vars=['RegionID', 'RegionName', 'SizeRank'], 
                             value_vars=date_columns_zhvi,
                             var_name='Date', 
                             value_name='HomeValue')

zri_melted = zri_data.melt(id_vars=['RegionID', 'RegionName', 'SizeRank', 'RegionType'], 
                           value_vars=date_columns_zri,
                           var_name='Date', 
                           value_name='RentValue')

market_report_melted = market_report_data.melt(id_vars=['RegionID', 'RegionName', 'SizeRank', 'RegionType'], 
                                               value_vars=date_columns_market,
                                               var_name='Date', 
                                               value_name='MarketMetric')  # Adjust 'MarketMetric' if necessary

new_construction_melted = new_construction_data.melt(id_vars=['RegionID', 'RegionName', 'SizeRank', 'RegionType'], 
                                                     value_vars=date_columns_construction,
                                                     var_name='Date', 
                                                     value_name='NewConstruction')  # Adjust column if necessary

# Step 2: Ensure the 'Date' columns are all in datetime format
zhvi_melted['Date'] = pd.to_datetime(zhvi_melted['Date'], errors='coerce')
zri_melted['Date'] = pd.to_datetime(zri_melted['Date'], errors='coerce')
market_report_melted['Date'] = pd.to_datetime(market_report_melted['Date'], errors='coerce')
new_construction_melted['Date'] = pd.to_datetime(new_construction_melted['Date'], errors='coerce')

# Step 3: Merge the datasets on 'RegionID', 'RegionName', and 'Date'
combined_data = pd.merge(zhvi_melted, zri_melted, on=['RegionID', 'RegionName', 'Date'], how='inner')
combined_data = pd.merge(combined_data, market_report_melted, on=['RegionID', 'RegionName', 'Date'], how='inner')

# Convert the Date column in the final dataset before the last merge
new_construction_melted['Date'] = pd.to_datetime(new_construction_melted['Date'], errors='coerce')
new_construction_melted = new_construction_melted.drop(columns=['SizeRank'])

# Final merge with new construction data
final_data = pd.merge(combined_data, new_construction_melted, on=['RegionID', 'RegionName', 'Date'], how='inner')

# Preview the final merged dataset
print(final_data.head())
