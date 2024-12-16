import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file (only the first 100,000 rows)
file_path = "KDDTrain.csv"  # Replace with the actual file path
data = pd.read_csv(file_path, header=None, nrows=100000)

# Select the second, third, and fourth columns
columns_to_encode = data.iloc[:, 1:4]

# Use LabelEncoder to encode each column
label_encoders = {}
for col in columns_to_encode.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Modify data[col] directly
    label_encoders[col] = le  # Store the LabelEncoder object
    print(2)

# Output the transformed data (you can print a portion for inspection)
print(data.head())

# Save to a new file
data.to_csv("KDDdata.csv", index=False, header=False)
