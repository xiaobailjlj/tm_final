import pandas as pd

# Load the CSV file
file_path = './news_bias_dataset/preprocessed_dataset.csv'
data = pd.read_csv(file_path)

# Get the count of each unique score in the 'bias_score' column
bias_score_counts = data['bias_score'].value_counts()

# Print the counts
print("Count of Each Bias Score:")
print(bias_score_counts)