from datasets import load_dataset, DatasetDict
import json
import os
import pandas as pd

def load_and_format_sustainability_data(file_path, min_length=50):
    if os.path.exists(file_path) and file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            formatted_data = [
                {'text': f"Question: {qa['question'].strip()} \nAnswer: {qa['answer'].strip()}"}
                for qa in data if len(qa['question'].strip()) >= min_length and len(qa['answer'].strip()) >= min_length
            ]
            return pd.DataFrame(formatted_data)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()


def get_formatted_data_hf(dataset_name, question_key, answer_key):
    try:
        # Attempt to load the dataset
        dataset = load_dataset(dataset_name, split="train", download_mode="force_redownload")
        
        # Log dataset metadata
        if isinstance(dataset, DatasetDict):
            print(f"Loaded a DatasetDict with splits: {list(dataset.keys())}")
            dataset = dataset["train"]  # Assuming 'train' split is what we need

        print(f"Dataset {dataset_name} loaded with {len(dataset)} examples.")

        # Check if expected keys are in the dataset
        if question_key not in dataset.column_names or answer_key not in dataset.column_names:
            raise ValueError(f"Dataset must contain columns '{question_key}' and '{answer_key}'")

        # Format the data
        formatted_data = [
            {'text': f"Question: {item[question_key].strip()} \nAnswer: {item[answer_key].strip()}"}
            for item in dataset if question_key in item and answer_key in item
        ]
        print(f"Formatted data with {len(formatted_data)} entries.")

        return pd.DataFrame(formatted_data)
    except Exception as e:
        # Catch any type of Exception and log detailed information
        print(f"Failed to load or format {dataset_name}: {str(e)}")


def get_formatted_slim_orca_data():
    try:
        # Load the dataset
        dataset = load_dataset("Open-Orca/SlimOrca", split="train", download_mode="force_redownload")
        formatted_data = []

        # Iterate over each conversation in the dataset
        for item in dataset:
            conversation = item['conversations']
            question = ""
            answer = ""

            # Extract the question and answer from the conversation
            for turn in conversation:
                if turn['from'] == "human":
                    question = turn['value'].strip()
                elif turn['from'] == "gpt":
                    answer = turn['value'].strip()

            # If both question and answer are found, format and add to list
            if question and answer:
                formatted_data.append({
                    'text': f"Question: {question} \nAnswer: {answer}"
                })

        # Convert the list to a DataFrame
        return pd.DataFrame(formatted_data)
    except Exception as e:
        print(f"Failed to load or format Open-Orca/SlimOrca: {e}")
        return pd.DataFrame()


def get_formatted_xsum_data():
    try:
        dataset = load_dataset("EdinburghNLP/xsum", split="train", download_mode="force_redownload")
        formatted_data = [
            {'text': f"Question: You are an AI assistant. You need to make a concise summary from the following text: {item['document'].strip()} \nAnswer: {item['summary'].strip()}"}
            for item in dataset
        ]
        return pd.DataFrame(formatted_data)
    except Exception as e:
        print(f"Failed to load or format xsum dataset: {e}")
        return pd.DataFrame()


def save_to_json(output_path, data):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Failed to save data to {output_path}: {e}")


def main():
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'sustainability_pdfs', 'results')
    os.makedirs(results_dir, exist_ok=True)

    df_sustainability = load_and_format_sustainability_data(os.path.join(results_dir, 'final_results.json'), min_length=10)
    print("Sustainability data loaded and formatted successfully.")

    # Process and format the data
    df_slim_orca = get_formatted_slim_orca_data()
    print("SlimOrca data loaded and formatted successfully.")

    df_wyvern = get_formatted_data_hf("StudentLLM/Open-Wyvern-74k", 'instruction', 'response')
    print("Wyvern data loaded and formatted successfully.")
    
    df_xsum = get_formatted_xsum_data()
    print("XSum data loaded and formatted successfully.")

    final_dataset = pd.concat([df_sustainability, df_slim_orca, df_wyvern, df_xsum], ignore_index=True)
    save_to_json(os.path.join(results_dir, 'all_data_combined.json'), final_dataset.to_dict(orient='records'))
    print("All data has been combined and saved successfully.")

if __name__ == "__main__":
    main()