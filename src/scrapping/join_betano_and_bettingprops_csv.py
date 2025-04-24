import pandas as pd
import os

# Get the current working directory and go up one level to the project root
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Read the first CSV file
betting_data_path = os.path.join(current_dir, 'betting_data_2025-04-21.csv')
betting_data = pd.read_csv(betting_data_path)

# Remove rows where Over Line or Under Line is empty
betting_data = betting_data.dropna(subset=['Over Line', 'Under Line'])

# Remove the specified columns from betting_data - only those that exist
columns_to_drop = ['Over Line', 'Under Line']
# Only drop columns that actually exist in the dataframe
columns_to_drop = [col for col in columns_to_drop if col in betting_data.columns]
betting_data = betting_data.drop(columns=columns_to_drop)

# Read the second CSV file
betano_data_path = os.path.join(current_dir, 'betano_strikeouts_20250421_133706.csv')
betano_data = pd.read_csv(betano_data_path)

# Remove the Team column from betano_data
betano_data = betano_data.drop(columns=['Team'])

# Perform the join on the Player column
# Using left join to keep all records from betting_data
merged_data = pd.merge(betting_data, betano_data, on='Player', how='left')

# Remove rows where the Line column is empty
merged_data = merged_data.dropna(subset=['Line'])

# Save back to the original betting_data file
merged_data.to_csv(betting_data_path, index=False)

print(f"Files merged successfully. Original file has been updated: {betting_data_path}")
print(f"Total rows in merged data: {len(merged_data)}")
