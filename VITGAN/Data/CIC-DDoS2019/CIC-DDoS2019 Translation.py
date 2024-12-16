import os
import pandas as pd
import numpy as np

# Initialize an empty DataFrame to merge all data
all_data = pd.DataFrame()

# Define the directory storing the CSV files
data_folder = ''

# Get the paths of all CSV files
files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

# Iterate through each file for data processing
for file in files:
    # Read the total number of rows in the CSV file
    with open(file, 'r') as f:
        total_rows = sum(1 for line in f)  # Total number of rows in the file

    # Randomly generate the rows to skip
    skip_rows = np.random.choice(
        range(1, total_rows),  # From the 1st row to the last row
        replace=False,
        size=max(0, total_rows - np.random.randint(3000, 15001))  # Keep between 3000 and 15000 rows
    )

    # Read the file and skip the specified rows
    temp_df = pd.read_csv(file, skiprows=skip_rows)

    # Add a new column indicating the source file, representing the attack type
    temp_df['Attack Type'] = os.path.basename(file).replace('.csv', '')

    # Merge data into all_data
    all_data = pd.concat([all_data, temp_df], ignore_index=True)

# Clean up column names, remove leading/trailing spaces
all_data.columns = all_data.columns.str.strip()

# Replace infinite values with NaN and fill missing values with the median
all_data = all_data.replace([np.inf, -np.inf], np.nan)
all_data.fillna(all_data.median(numeric_only=True), inplace=True)

# Save the processed data to the current directory
output_file = './CIC19datadata.csv'
all_data.to_csv(output_file, index=False)

print(f"All data has been processed and saved to {output_file}")
