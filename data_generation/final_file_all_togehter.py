from datasets import load_dataset, load_from_disk
import json
import os
import pandas as pd

def load_and_filter_sustainability_data(directory, min_length=50):
    all_qa_pairs = []  
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:  
                data = json.load(file)
                for entry in data.values(): 
                    for qa in entry:
                        question = qa['question'].strip()
                        answer = qa['answer'].strip()
                        # Filter by minimum length
                        if len(question) >= min_length and len(answer) >= min_length:
                            all_qa_pairs.append({'question': question, 'answer': answer})
    return all_qa_pairs

def format_row(row):
    question = row['question']
    answer = row['answer']
    formatted = f"[INST] {question} [/INST] {answer}"
    return formatted

def get_format_data_hf(dataset_name):
    dataset = load_dataset(dataset_name, split="train")
    df = pd.DataFrame(dataset)
    df['formatted'] = df.apply(format_row, axis=1)
    df = df[['formatted']].rename(columns={'formatted': 'text'})
    return df

def get_format_data_json(dataset_name):
    with open(dataset_name, encoding="utf8") as file:
        data = json.load(file)
    formatted_data = [f"[INST] {item['question']} [/INST] {item['answer']}" for item in data]
    df = pd.DataFrame(formatted_data, columns=['text'])
    return df

def save_to_json(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'pdfs', 'sustainability', 'results')
    formatted_data = [f"[INST] {item['question']} [/INST] {item['answer']}" for item in qa_pairs]

    # LOADING SUSATAINABILITY QA DATA
    qa_pairs = load_and_filter_sustainability_data(results_dir, min_length=10)
    final_output_path = os.path.join(results_dir, 'final', 'final_results.json')
    save_to_json(final_output_path, qa_pairs)
    print(f"Filtered data saved to {final_output_path}")

    df_sustainability_qa = pd.DataFrame(formatted_data, columns=['text'])
    print(f"Sustainability QA data formated, the amount of rows is: {len(df_sustainability_qa)}")

    # LOADING GENERAL QA DATA
    df_general_qa = get_format_data_hf("Open-Orca/OpenOrca")
    print("General QA data loaded and formatted, the amount of rows is: ", len(df_general_qa))
    df_general_qa_2 = get_format_data_json("data_generation/general_qa.json")



    # LOADING SUMMARIES QA DATA
    df_summary_qa = get_format_data_hf("EdinburghNLP/xsum")
    print("Summaries QA data loaded and formatted, the amount of rows is: ", len(df_summary_qa))


if __name__ == "__main__":
    main()
