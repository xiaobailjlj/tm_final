import pandas as pd
import re

# Load the dataset (adjust file path as needed)
file_path = "news_bias_dataset/Sora_LREC2020_biasedsentences.csv"
data = pd.read_csv(file_path)

# Define the columns to extract
columns_to_keep = ["id_event", "id_article", "article_bias"]
sentence_ids = ["t", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
text_columns = ["doctitle", "s0", "s1", "s2", "s3", "s4", "s5", "s6",
                "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
                "s15", "s16", "s17", "s18", "s19"]

source_bias_dict = {"left": 2, "left-center": 1, "least": 0, "right-center": 1, "right": 2, "right-extream": 3}

# Initialize a list to store the rows for the final DataFrame
processed_rows = []

# Process each row individually
for _, row in data.iterrows():
    for id_sentence, text_col in zip(sentence_ids, text_columns):
        # Append a new row if the sentence text is not null
        sentence_text = row[text_col]
        bias_score = row[id_sentence]
        source_bias = source_bias_dict[row["source_bias"]]
        if pd.notnull(sentence_text):
            cleaned_text = re.sub(r"^\[\d+\]:\s*", "", sentence_text)
            processed_rows.append({
                "id_event": row["id_event"],
                "source_bias": source_bias,
                "id_article": row["id_article"],
                "article_bias": row["article_bias"],
                "id_sentence": id_sentence,
                "sentence_text": cleaned_text,
                "bias_score": bias_score - 1  # Convert to 0-based index
            })

# Convert the list of processed rows into a DataFrame
processed_data = pd.DataFrame(processed_rows)

# Save to CSV
processed_data.to_csv("news_bias_dataset/preprocessed_dataset.csv", index=False)

print(processed_data.head())
