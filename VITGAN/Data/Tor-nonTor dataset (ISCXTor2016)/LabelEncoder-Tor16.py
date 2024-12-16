import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read CSV file (only the first 80,000 rows)
file_path = "tor_identification.csv"  # Replace with the actual file path
data = pd.read_csv(file_path, header=None, nrows=80000)

# Select the first, third, and eighth columns
columns_to_encode = data.iloc[:, [0, 2, 7]]

# Use LabelEncoder to convert each selected column into numeric values
label_encoders = {}
for col in columns_to_encode.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Modify data[col] in place
    label_encoders[col] = le  # Store the LabelEncoder object
    print(2)

# Output the transformed data (print a portion for verification)
print(data.head())

# Save the transformed data to a new file
data.to_csv("Tor16data.csv", index=False, header=False)
