import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for pandas
pd.set_option("display.max_columns", None)

# Define plot styling parameters
large = 22;
med = 16;
small = 12
params = {'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
sns.set_style("whitegrid")

# Load the dataset
file_path = 'data_capec_multilabel.csv'  # Ensure this path is correct
dataset = pd.read_csv(file_path, low_memory=False)

# Function to compute mean ASCII encoding for a string
def mean_ascii_encoding(value):
    if type(value) != str and np.isnan(value):
        result = 0  # Return 0 for NaN values
    else:
        text = str(value).lower()  # Convert to lowercase string
        L = len(text)  # Length of string
        if L == 0:
            result = 0  # Return 0 if the string is empty
        else:
            v = 0  # Initialize ASCII accumulator
            for char in text:  # Traverse each character in the string
                v += ord(char)  # Accumulate ASCII value of each character
            result = v / L  # Return the average ASCII value

    return result

# Vectorize the mean_ascii_encoding function for efficient array operations
mean_ascii_encoding = np.vectorize(mean_ascii_encoding)

# Extract features and labels
features_raw = dataset[dataset.columns[5:24]].to_numpy()  # Extract features (columns 6 to 24)
labels = dataset[dataset.columns[24:]].to_numpy()  # Extract labels (columns 25 onward)

# Apply mean ASCII encoding to features
features_values = mean_ascii_encoding(features_raw)

# Combine processed features and labels into a single DataFrame
data = pd.DataFrame(data=np.concatenate((features_values, labels), axis=1),
                    columns=dataset.columns[5:])

# Convert appropriate columns to correct data types
data = data.astype(dict(zip(dataset.columns[5:], [np.float32] * 19 + [np.int8] * 13)))

# Create a DataFrame to summarize the number of requests per label
stat = pd.DataFrame(data=np.sum(labels, axis=0), index=data.columns[19:], columns=['Number of web requests'])

# Calculate the percentage of total requests for each label
stat['% of total requests'] = stat['Number of web requests'] / len(data) * 100

# Calculate unique counts of the sum of labels for each request
unique_counts = np.unique(np.sum(labels, axis=1), return_counts=True)

# Save the processed data to a new CSV file
processed_file_path = 'SRBHdata.csv'  # Specify the path and file name to save the new CSV file
data.to_csv(processed_file_path, index=False)

# Optionally, save the label statistics as well
stat_file_path = 'label_statistics.csv'  # Path for the statistics file
stat.to_csv(stat_file_path)

print(f"Processed data saved to: {processed_file_path}")
print(f"Label statistics saved to: {stat_file_path}")
