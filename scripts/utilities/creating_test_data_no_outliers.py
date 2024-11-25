import json
import numpy as np
from transformers import AutoTokenizer
import os

def analyze_and_preprocess_data(tokenizer_name, file_path, output_file_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load and tokenize the data to analyze token lengths
    lengths = []
    data_entries = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            input_text = data['text']  # Adjust based on your data structure
            tokens = tokenizer.encode(input_text, add_special_tokens=True)
            lengths.append(len(tokens))
            data_entries.append(data)  # Store the whole data object for optional filtering later

    # Analyzing token length distribution
    print("Token length analysis for:", tokenizer_name)
    print("Mean length:", np.mean(lengths))
    print("Median length:", np.median(lengths))
    print("80th percentile:", np.percentile(lengths, 80))

    # Deciding on a conservative max_length based on the analysis
    max_length = int(np.percentile(lengths, 80))  # or any other percentile or fixed value you deem appropriate

    # Filtering data that exceeds the determined max_length
    processed_data = [data for data, length in zip(data_entries, lengths) if length <= max_length]

    # Write the processed data back to a new file, each JSON object on its own line
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            json_string = json.dumps(item, ensure_ascii=False)
            outfile.write(json_string + '\n')  # Write each JSON object on a new line
    
    print(f"Filtered data written to {output_file_path}. Used max_length: {max_length}")


pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
results_dir = os.path.join(pdf_dir, 'results')
file_path = os.path.join(results_dir, 'test_data_llama3.json')
output_file_path = os.path.join(results_dir, 'test_data_llama3_no_outliers.json')

# Parameters
tokenizer_name = "meta-llama/Meta-Llama-3-8B"  # or "llama2-13b-specific-tokenizer"

# Running the function
analyze_and_preprocess_data(tokenizer_name, file_path, output_file_path)
