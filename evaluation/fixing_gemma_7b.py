import json

def load_processed_questions(results_file):
    """Load the questions that have already been processed from the results file."""
    processed_questions = set()
    with open(results_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                if line.strip():
                    data = json.loads(line)
                    processed_questions.add(data['question'])
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_number}: {line}")
    return processed_questions

def filter_unprocessed_questions(original_data_file, processed_questions):
    """Filter out questions that have already been processed."""
    unprocessed_data = []
    with open(original_data_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                if line.strip():
                    data = json.loads(line)
                    question = data['text'].split("\nOutput:")[0].strip()
                    if question not in processed_questions:
                        unprocessed_data.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_number}: {line}")
    return unprocessed_data

def write_unprocessed_data(unprocessed_data, output_file):
    """Write the unprocessed questions to a new JSON file with UTF-8 encoding directly."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for data in unprocessed_data:
            json.dump(data, file, ensure_ascii=False)  # Set ensure_ascii to False to write non-ASCII characters directly
            file.write('\n')

def main():
    results_file = './results/gemma-7b_automatic_evaluation.json'
    original_data_file = '../data/final/test_data_llama2-13b.json'
    output_file = './results/test_data_llama2-13b_unprocessed_gemma7b.json'

    # Load processed questions
    processed_questions = load_processed_questions(results_file)

    # Filter unprocessed questions from the original dataset
    unprocessed_data = filter_unprocessed_questions(original_data_file, processed_questions)

    # Write the unprocessed questions to a new file
    write_unprocessed_data(unprocessed_data, output_file)

    print(f"Filtered unprocessed data written to {output_file}. Total unprocessed items: {len(unprocessed_data)}")

if __name__ == "__main__":
    main()
