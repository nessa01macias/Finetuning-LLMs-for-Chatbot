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

def generate_predictions(model, tokenizer, questions, batch_size=16):
    predictions = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs, max_length=512)
        batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predictions.extend(batch_predictions)
    return predictions

def calculate_bleu(predictions, references):
    bleu_metric = load_metric("bleu")
    results = bleu_metric.compute(predictions=[pred.split() for pred in predictions], references=references)
    return results['bleu']

def save_results(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    test_data_path = '/scratch/project_2008167/thesis/data/test_data.json'
    results_path = '/scratch/project_2008167/thesis/evaluation/phi-2_automatic_evaluation.json'
    
    # Load model and tokenizer from Hugging Face
    model = AutoModelForCausalLM.from_pretrained("nessa01macias/phi-2_sustainability-qa", trust_remote_code=False, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    questions, references = load_test_data(test_data_path)

    # Generate predictions
    predictions = generate_predictions(model, tokenizer, questions, batch_size=16)

    # Calculate BLEU score
    bleu_score = calculate_bleu(predictions, references)
    print("BLEU Score:", bleu_score)

        # Prepare data for saving
    results_data = [{
        "question": q,
        "generated_answer": pred,
        "reference_answer": ref[0]
    } for q, pred, ref in zip(questions, predictions, references)]

    # Save results
    save_results(results_path, results_data)


if __name__ == "__main__":
    main()
