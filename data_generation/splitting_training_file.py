import json
import os
import pandas as pd
from sklearn.utils import shuffle

def main():
    # Path setup for the JSON file containing the data
    pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
    combined_data_path = os.path.join(pdf_dir, 'final', 'all_data_combined.json')
    results_dir = os.path.join(pdf_dir,  'final')

    # Load and shuffle the JSON data
    with open(combined_data_path, encoding="utf8") as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    print("The all_data_combined length is: ", len(df))
    df = shuffle(df, random_state=42)

    # Split the DataFrame into train and test sets
    train_size = int(0.75 * len(df))
    train_df = df[:train_size]
    print("The train_data length is: ", len(train_df))
    test_df = df[train_size:]
    print("The test_data length is: ", len(test_df))

    # Save the train and test sets to JSON
    train_file_path = os.path.join(results_dir, 'train_data.json')
    test_file_path = os.path.join(results_dir, 'test_data.json')
    train_df.to_json(train_file_path, orient='records', lines=True, force_ascii=False)
    test_df.to_json(test_file_path, orient='records', lines=True, force_ascii=False)
    print(f"Data has been split and saved.\nTrain data: {train_file_path}\nTest data: {test_file_path}")

    # Split the train data into 10 parts
    split_json_file(train_file_path, results_dir, 5)

def split_json_file(source_file_path, output_directory, number_of_splits):
    # Read the entire data from the source file
    with open(source_file_path, 'r', encoding='utf8') as file:
        data = [json.loads(line) for line in file]

    # Calculate the size of each split
    total_entries = len(data)
    split_size = total_entries // number_of_splits
    print(f"Total entries: {total_entries}")
    print(f"Entries per split: {split_size}")

    # Split the data and write to new files
    for i in range(number_of_splits):
        start_index = i * split_size
        end_index = start_index + split_size if i != number_of_splits - 1 else total_entries
        split_data = data[start_index:end_index]
        print(f"Split {i+1} has {len(split_data)} entries")
        split_file_path = os.path.join(output_directory, f"train_data_split_{i+1}.json")
        with open(split_file_path, 'w', encoding='utf8') as split_file:
            json.dump(split_data, split_file, ensure_ascii=False, indent=4)
        print(f"Written split {i+1} to {split_file_path}")

if __name__ == "__main__":
    main()
