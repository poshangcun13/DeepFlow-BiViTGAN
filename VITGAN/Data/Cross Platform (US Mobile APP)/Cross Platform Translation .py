import os
import json
import csv

# Define the field names
fields = [
    "timestamp", "domain", "dst_ip", "dst_port", "connection", "content_type", "host",
    "is_host_ip", "label", "md5", "method", "package_name", "pii_types", "platform",
    "post_body", "referrer", "src_ip", "src_port", "tk_flag", "uri", "user_agent"
]

# List of folder paths
folders = [ r"data\anubis", r"data\auto",r"data\manual\android",]

# Output CSV file path
output_csv = r"appdata.csv"

# Initialize a list to store the results
data = []

# Traverse all JSON files in the folders
for folder in folders:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        # Read the JSON data
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            # Extract the required fields
                            row = {field: None for field in fields}
                            row["timestamp"] = key.split(",")[0]

                            # Extract fields from the JSON data
                            for field in fields[1:]:
                                if field in value:
                                    row[field] = value[field]

                            # Process the 'headers' field and merge it into the row
                            if "headers" in value:
                                headers = value["headers"]
                                for header_key, header_value in headers.items():
                                    header_field = header_key.replace("-", "_").lower()  # Convert the field name to CSV format
                                    if header_field in row:
                                        row[header_field] = header_value

                            # Add the processed row to the data list
                            data.append(row)
                    except json.JSONDecodeError as e:
                        print(f"Error reading {file_path}: {e}")

# Write the data to a CSV file
with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data)

print(f"All data has been successfully extracted to {output_csv}")
