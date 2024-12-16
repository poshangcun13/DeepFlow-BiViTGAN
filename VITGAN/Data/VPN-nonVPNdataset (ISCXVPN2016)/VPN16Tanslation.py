import pandas as pd
from scipy.io import arff
import numpy as np

# List of input ARFF file paths
input_arff_files = [
    "TimeBasedFeatures-Dataset-15s-VPN.arff",
    "TimeBasedFeatures-Dataset-30s-VPN.arff",
    "TimeBasedFeatures-Dataset-120s-VPN.arff"
]
output_csv_arff = "TimeBasedFeatures-Dataset-Merged.csv"  # Output merged CSV file path

# Create an empty list to store DataFrames
dataframes = []

# Read and process each ARFF file
for input_arff in input_arff_files:
    # Read ARFF file
    data, meta = arff.loadarff(input_arff)
    df = pd.DataFrame(data)

    # Handle empty strings by replacing them with NaN
    df.replace("", np.nan, inplace=True)

    # Decode byte columns to normal strings
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':  # Ensure it's byte data
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Convert label column to 1 for 'benign' and 0 for 'vpn'
    if 'label' in df.columns:  # Ensure 'label' column exists
        df['label'] = df['label'].replace({'benign': 1, 'vpn': 0})

    # Add DataFrame to list
    dataframes.append(df)

# Merge all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Save to CSV file
merged_df.to_csv(output_csv_arff, index=False, encoding='utf-8')
print(f"All files have been merged and saved as {output_csv_arff}")

# Process multiple CSV files
csv_files = [
    "TimeBasedFeatures-Dataset-15s-VPN.csv",
    "TimeBasedFeatures-Dataset-30s-VPN.csv",
    "TimeBasedFeatures-Dataset-120s-VPN.csv"
]

# Read the first file and preserve the header
df_combined = pd.read_csv(csv_files[0])

# From the second file onwards, read and concatenate, skipping the header
for file in csv_files[1:]:
    df_temp = pd.read_csv(file)
    df_combined = pd.concat([df_combined, df_temp], ignore_index=True)

# Save the combined CSV file
output_csv_combined = "vpn16.csv"
df_combined.to_csv(output_csv_combined, index=False, encoding='utf-8')

print(f"Files merged successfully and saved as {output_csv_combined}")
