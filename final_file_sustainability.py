import json
import os

def load_and_filter_data(directory, min_length=50):
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

def save_to_json(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f: 
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    base_dir = os.getcwd()
    results_dir = os.path.join(base_dir, 'sustainability_pdfs', 'results')
    final_output_path = os.path.join(results_dir, 'final', 'final_results.json')

    qa_pairs = load_and_filter_data(results_dir, min_length=10)  
    save_to_json(final_output_path, qa_pairs)
    print(f"Filtered data saved to {final_output_path}")

if __name__ == "__main__":
    main()
