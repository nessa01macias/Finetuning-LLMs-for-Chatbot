from datasets import load_dataset
import json
import os
import pandas as pd

def load_and_format_sustainability_data(file_path, min_length=50):
    try:
        # Load and format sustainability data from a JSON file.
        if os.path.exists(file_path) and file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                formatted_data = []
                for qa in data:
                    question, answer = qa['question'].strip(), qa['answer'].strip()
                    if len(question) >= min_length and len(answer) >= min_length:
                        formatted_data.append({'text': f"[INST] {question} [/INST] {answer}"})
                return pd.DataFrame(formatted_data)
        else:
            raise FileNotFoundError(f"No JSON file found at {file_path}")
    except Exception as e:
        print(f"Error loading or formatting sustainability data: {e}")
    return pd.DataFrame()


def get_formatted_data_hf(dataset_name, key_map):
    # Load and format data from Hugging Face datasets.
    try:
        dataset = load_dataset(dataset_name, split="train")
        df = pd.DataFrame(dataset)
        df['text'] = df.apply(lambda row: f"[INST] {row[key_map['question']]} [/INST] {row[key_map['answer']]}", axis=1)
        return df[['text']]
    except Exception as e:
        print(f"Error loading or formatting data from Hugging Face {dataset_name}: {e}")
        return pd.DataFrame()

def save_to_json(output_path, data):
    # Save data to a JSON file.
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving data to {output_path}: {e}")

def main():
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'sustainability_pdfs', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load and format local sustainability data.
    df_sustainability = load_and_format_sustainability_data(os.path.join(results_dir, 'final_results.json'), min_length=10)
    print("Sustainability data has been loaded and formatted successfully.")

    # Load and format data from Hugging Face datasets.
    df_orca = get_formatted_data_hf("Open-Orca/OpenOrca", {'question': 'question', 'answer': 'response'})
    print("Orca data has been loaded and formatted successfully.")
    df_wyvern = get_formatted_data_hf("StudentLLM/Open-Wyvern-74k", {'question': 'instruction', 'answer': 'response'})
    print("Wyvern data has been loaded and formatted successfully.")

    df_xsum = get_formatted_data_hf("EdinburghNLP/xsum", {'question': 'documents', 'answer': 'summary'})
    print("XSum data has been loaded and formatted successfully.")

    # Combine all datasets.
    final_dataset = pd.concat([df_sustainability, df_orca, df_wyvern, df_xsum], ignore_index=True)
    save_to_json(os.path.join(results_dir, 'all_data_combined.json'), final_dataset.to_dict(orient='records'))
    print("All data has been combined and saved successfully.")

if __name__ == "__main__":
    main()
