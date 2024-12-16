import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file (only the first 35,000 rows)
file_path = "appdata.csv"  # Replace with the actual file path
data = pd.read_csv(file_path, header=None, nrows=35000)

# Select the first, second, fourth, fifth, sixth, eighth, ninth, tenth, eleventh, twelfth, thirteenth, seventeenth, and eighteenth columns
columns_to_encode = data.iloc[:, [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 18]]

# Use LabelEncoder to transform each column into numerical values
label_encoders = {}
for col in columns_to_encode.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Directly modify data[col]
    label_encoders[col] = le  # Store the LabelEncoder object
    print(2)

# Output the transformed data (you can print part of it for inspection)
print(data.head())

# Save the transformed data to a new file
data.to_csv("Cross Platform data.csv", index=False, header=False)
