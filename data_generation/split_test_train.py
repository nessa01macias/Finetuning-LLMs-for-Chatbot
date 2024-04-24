import json
import os
import pandas as pd
from sklearn.utils import shuffle

# Path setup for the JSON file containing the data
pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
combined_data = os.path.join(pdf_dir, 'sustainability_pdfs', 'results', 'final', 'all_data_combined.json')
results_dir = os.path.join(pdf_dir, 'sustainability_pdfs', 'results', 'final')

# Load JSON data
with open(combined_data, encoding="utf8") as file:
    data = json.load(file)

# Convert data to DataFrame
df = pd.DataFrame(data)

# Shuffle the DataFrame
df = shuffle(df, random_state=42)

# Split the DataFrame into train and test sets
train_size = int(0.75 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Save the train and test sets to JSON
train_file_path = os.path.join(results_dir, 'train_data.json')
test_file_path = os.path.join(results_dir, 'test_data.json')

train_df.to_json(train_file_path, orient='records', lines=True, force_ascii=False)
test_df.to_json(test_file_path, orient='records', lines=True, force_ascii=False)

print(f"Data has been split and saved.\nTrain data: {train_file_path}\nTest data: {test_file_path}")
