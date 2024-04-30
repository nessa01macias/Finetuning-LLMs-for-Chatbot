import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric

def load_test_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    questions = [item['text'].split("\nAnswer:")[0].strip() for item in data]
    references = [[item['text'].split("\nAnswer:")[1].strip()] for item in data]
    return questions, references

def generate_and_save_predictions(model, tokenizer, questions, references, results_path, batch_size=8):
    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_references = references[i:i+batch_size]
            # Adjust max_length to accommodate longer contexts for Gemma 2b
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=8192)
            # You might want to adjust max_new_tokens depending on your specific requirements or keep it dynamic
            outputs = model.generate(**inputs, max_new_tokens=1024)  # Increased max_new_tokens for potentially longer outputs
            batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Save each batch of results immediately
            for question, prediction, reference in zip(batch_questions, batch_predictions, batch_references):
                result_data = {
                    "question": question,
                    "generated_answer": prediction,
                    "reference_answer": reference[0]
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

def calculate_bleu(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        predictions = []
        references = []
        for line in f:
            entry = json.loads(line)
            predictions.append(entry['generated_answer'].split())
            references.append([entry['reference_answer'].split()])
    bleu_metric = load_metric("bleu")
    results = bleu_metric.compute(predictions=predictions, references=references)
    return results['bleu']

def main():
    test_data_path = '/scratch/project_2008167/thesis/data/test_data.json'
    results_path = '/scratch/project_2008167/thesis/evaluation/gemma-7b_automatic_evaluation.json'
    
    # Load model and tokenizer from Hugging Face
    model = AutoModelForCausalLM.from_pretrained("nessa01macias/gemma-7b_sustainability-qa", trust_remote_code=False, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    questions, references = load_test_data(test_data_path)

    # Generate predictions and save them
    generate_and_save_predictions(model, tokenizer, questions, references, results_path, batch_size=8)

    # Calculate BLEU score after all data has been processed and saved
    bleu_score = calculate_bleu(results_path)
    print("BLEU Score:", bleu_score)

if __name__ == "__main__":
    main()
