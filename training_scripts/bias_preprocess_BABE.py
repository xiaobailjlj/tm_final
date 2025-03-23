import pandas as pd
import re

# Load the dataset (adjust file path as needed)
file_path = "news_bias_dataset/BABE/final_labels_all.csv"
data = pd.read_csv(file_path, sep=';')

# Define the columns to extract
columns_to_keep = ["text", "label_bias"]
data = data[columns_to_keep]

# Add the 'score' column based on conditions
data['bias_score'] = data['label_bias'].apply(lambda x: 0 if x == "Non-biased" else 1 if x == "Biased" else None)

# Drop rows where 'score' is None (other values of 'label_bias')
data = data.dropna(subset=['bias_score'])

# Convert 'score' to integer type for consistency
data['bias_score'] = data['bias_score'].astype(int)

# Display the first few rows to confirm
print(data.head())

# Save the processed data to a new CSV file
output_file_path = "news_bias_dataset/BABE/processed_labels.csv"
data.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")