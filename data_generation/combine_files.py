from datasets import load_dataset
import json
import os
import pandas as pd

def load_and_format_sustainability_data(file_path, min_length=50):
    if os.path.exists(file_path) and file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            formatted_data = [
                {'text': f"[INST] {qa['question'].strip()} [/INST] {qa['answer'].strip()}"}
                for qa in data if len(qa['question'].strip()) >= min_length and len(qa['answer'].strip()) >= min_length
            ]
            return pd.DataFrame(formatted_data)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

def get_formatted_data_hf(dataset_name, question_key, answer_key):
    try:
        dataset = load_dataset(dataset_name, split="train")
        formatted_data = [
            {'text': f"[INST] {item[question_key].strip()} [/INST] {item[answer_key].strip()}"}
            for item in dataset if question_key in item and answer_key in item
        ]
        return pd.DataFrame(formatted_data)
    except Exception as e:
        print(f"Failed to load or format {dataset_name}: {e}")
        return pd.DataFrame()

def get_formatted_xsum_data():
    try:
        dataset = load_dataset("EdinburghNLP/xsum", split="train")
        formatted_data = [
            {'text': f"[INST] You are an AI assistant. You need to make a concise summary from the following text: {item['document'].strip()} [/INST] {item['summary'].strip()}"}
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
    df_orca = get_formatted_data_hf("Open-Orca/OpenOrca", 'question', 'response')
    df_wyvern = get_formatted_data_hf("StudentLLM/Open-Wyvern-74k", 'instruction', 'response')
    df_xsum = get_formatted_xsum_data()

    final_dataset = pd.concat([df_sustainability, df_orca, df_wyvern, df_xsum], ignore_index=True)
    save_to_json(os.path.join(results_dir, 'all_data_combined.json'), final_dataset.to_dict(orient='records'))
    print("All data has been combined and saved successfully.")

if __name__ == "__main__":
    main()
