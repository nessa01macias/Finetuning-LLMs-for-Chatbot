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
    device = model.device  # Get the device model is on
    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_references = references[i:i+batch_size]
            
            # Tokenize the questions and ensure the tensors are sent to the same device as the model
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device

            # Generate responses using the model
            outputs = model.generate(**inputs, max_new_tokens=548)  # Adjust max_new_tokens for your needs

            # Decode generated token ids to text
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

    torch.cuda.empty_cache()  # Clear any cached memory

    test_data_path = '/scratch/project_2008167/thesis/data/test_data.json'
    results_path = '/scratch/project_2008167/thesis/evaluation/gemma-2b_automatic_evaluation.json'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer from Hugging Face
    model = AutoModelForCausalLM.from_pretrained("nessa01macias/gemma-2b_sustainability-qa", trust_remote_code=False, torch_dtype=torch.float32, device_map = 'auto')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", trust_remote_code=False)
    model.to(device)

    print("Model and tokenizer loaded")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test data
    questions, references = load_test_data(test_data_path)

    # Generate predictions and save them
    generate_and_save_predictions(model, tokenizer, questions, references, results_path, batch_size=2)

    # Calculate BLEU score after all data has been processed and saved
    bleu_score = calculate_bleu(results_path)
    print("BLEU Score:", bleu_score)

if __name__ == "__main__":
    main()
