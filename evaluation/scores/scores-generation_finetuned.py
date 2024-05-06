from datasets import load_metric
import json
import os

def tokenize(text):
    """Tokenize the text by splitting on whitespace."""
    return text.split()

def process_entry(data):
    """Process an entry to format it correctly for metric computation."""
    try:
        index = data['index']
        question = data['question'].strip()
        generated_answer = data['generated_answer'].strip()
        reference_answer = data['reference_answer'].strip()

        if not generated_answer or reference_answer == "":
            print(f"Skipping due to empty reference answer at index {index}")
            return None

        return {
            'index': index,
            'question': question,
            'generated_answer': [tokenize(generated_answer)],
            'reference_answer': [[tokenize(reference_answer)]],
        }
    except KeyError as e:
        print(f"Missing key in data: {e}, index might be {data.get('index', 'Unknown')}")
        return None


def calculate_metrics(input_file, output_file):
    bleu_metric = load_metric('bleu')
    rouge_metric = load_metric('rouge')
    meteor_metric = load_metric('meteor')

    results = []
    all_predictions = []
    all_references = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            processed_data = process_entry(data)
            if not processed_data:
                continue

            # Compute individual sentence metrics
            bleu_score = bleu_metric.compute(predictions=processed_data['generated_answer'], references=processed_data['reference_answer'])['bleu']
            rouge_score = rouge_metric.compute(predictions=[' '.join(pred) for pred in processed_data['generated_answer']], references=[' '.join(ref) for ref in processed_data['reference_answer'][0]])['rougeL'].mid.fmeasure
            meteor_score = meteor_metric.compute(predictions=[' '.join(pred) for pred in processed_data['generated_answer']], references=[' '.join(ref) for ref in processed_data['reference_answer'][0]])['meteor']

            results.append({
                'index': processed_data['index'],
                'question': processed_data['question'],
                'reference': ' '.join(processed_data['reference_answer'][0][0]),
                'predicted': ' '.join(processed_data['generated_answer'][0]),
                'bleu': bleu_score,
                'rouge': rouge_score,
                'meteor': meteor_score
            })

            all_predictions.extend(processed_data['generated_answer'])
            all_references.extend(processed_data['reference_answer'])

    # Calculate overall metrics
    overall_bleu = bleu_metric.compute(predictions=all_predictions, references=all_references)['bleu']
    overall_rouge = rouge_metric.compute(predictions=[' '.join(pred) for pred in all_predictions], references=[' '.join(ref[0]) for ref in all_references])['rougeL'].mid.fmeasure
    overall_meteor = meteor_metric.compute(predictions=[' '.join(pred) for pred in all_predictions], references=[' '.join(ref[0]) for ref in all_references])['meteor']

    results.append({
        'overall_bleu': overall_bleu,
        'overall_rouge': overall_rouge,
        'overall_meteor': overall_meteor
    })

    # Write results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def process_all_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith("_automatic_evaluation_cleaned.json"):
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, filename.replace("_automatic_evaluation_cleaned.json", "_results.json"))
            print(f"Processing {input_file}...")
            calculate_metrics(input_file, output_file)
            print(f"Results written to {output_file}")

# Set the directory containing your files
current_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
directory_path = os.path.join(current_dir, 'finetuned')
process_all_files(directory_path)

