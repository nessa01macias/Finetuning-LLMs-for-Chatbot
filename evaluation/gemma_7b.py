import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_metric
import os
import time


def load_test_data(filepath):
    start_time = time.perf_counter()  # Start timing

    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    questions = [item['text'].split("\nAnswer:")[0].strip() for item in data]
    references = [[item['text'].split("\nAnswer:")[1].strip()] for item in data]

    end_time = time.perf_counter()  # End timing
    print(f"Loading test data took {end_time - start_time:.4f} seconds")
    return questions, references


def generate_and_save_predictions(model, tokenizer, questions, references, results_path, batch_size=8):
    start_time = time.perf_counter()  # Start timing

    device = model.device  # Get the device model is on
    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_references = references[i:i+batch_size]
            
            # Tokenize the questions and ensure the tensors are sent to the same device as the model
            inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=661)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move input tensors to the device

            start_time_inference = time.perf_counter()  # Start timing for inference
            # Generate responses using the model
            outputs = model.generate(**inputs, max_new_tokens=100)  # Adjust max_new_tokens for your needs
            end_time_inference = time.perf_counter()  # End timing for inference
            print(f"Inference for batch {i//batch_size+1}: took {start_time_inference - end_time_inference:.4f} seconds")

            # Decode generated token ids to text
            batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            # Clean up the generated answers by removing the repeated question
            cleaned_predictions = []
            for question, prediction in zip(batch_questions, batch_predictions):
                if prediction.startswith(question):
                    # Cut off the question part from the prediction
                    start_idx = len(question)
                    clean_prediction = prediction[start_idx:].strip()
                else:
                    clean_prediction = prediction
                cleaned_predictions.append(clean_prediction)

            # Save each batch of results immediately
            for question, prediction, reference in zip(batch_questions, cleaned_predictions, batch_references):
                result_data = {
                    "question": question,
                    "generated_answer": prediction,
                    "reference_answer": reference[0]
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

    end_time = time.perf_counter()  # End timing
    print(f"Generating and saving predictions took {end_time - start_time:.4f} seconds")


def calculate_bleu(filepath):

    start_time = time.perf_counter()  # Start timing

    with open(filepath, 'r', encoding='utf-8') as f:
        predictions = []
        references = []
        for line in f:
            entry = json.loads(line)
            predictions.append(entry['generated_answer'].split())
            references.append([entry['reference_answer'].split()])
    bleu_metric = load_metric("bleu")
    results = bleu_metric.compute(predictions=predictions, references=references)

    end_time = time.perf_counter()  # End timing
    print(f"Generating bleu metrics took {end_time - start_time:.4f} seconds")

    return results['bleu']

def main():
    start_time = time.perf_counter()  # Start timing for the entire main function

    # Environment configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.cuda.empty_cache()  # Clear any cached memory
    torch.backends.cudnn.deterministic = True  # Optional: for reproducibility

    test_data_path = '/scratch/project_2008167/thesis/data/test_data_llama2-13b.json'
    results_path = '/scratch/project_2008167/thesis/evaluation/base_gemma-7b_automatic_evaluation.json'
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    try:
        # Load model and tokenizer from Hugging Face
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=False)
        model.to(device)

        print("Model and tokenizer loaded - Gemma7b")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load test data
        questions, references = load_test_data(test_data_path)

        print("Test data loaded - Gemma7b")

        # Generate predictions and save them
        generate_and_save_predictions(model, tokenizer, questions, references, results_path, batch_size=4)


        # Calculate BLEU score after all data has been processed and saved
        bleu_score = calculate_bleu(results_path)
        print("BLEU Score - Gemma7b:", bleu_score)
    
    except Exception as e:
        print(f"An error occurred: {e}")

    end_time = time.perf_counter()  # End timing for the entire main function
    print(f"Total execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
