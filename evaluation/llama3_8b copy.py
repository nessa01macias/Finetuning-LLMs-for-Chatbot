import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the prompt structure used during training
def format_prompt(instruction, input_text=""):
    return f"""Below is an instruction that describes a task or a question, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def load_test_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    formatted_data = [{
        'prompt': format_prompt(item['instruction'], item.get('input', '')),
        'reference': item.get('output', '').strip()  # assuming the output is the desired response
    } for item in data if item.get('output', '').strip()]
    return formatted_data

def generate_and_save_predictions(model, tokenizer, data, results_path, batch_size=4):
    device = model.device
    model.to(device)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            prompts = [item['prompt'] for item in batch]
            references = [item['reference'] for item in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for prompt, prediction, reference in zip(prompts, predictions, references):
                result_data = {
                    "prompt": prompt,
                    "generated_answer": prediction,
                    "reference_answer": reference
                }
                f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

def main():
    # Load model and tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("nessa01macias/llama3-8b_sustainability-qa-ins", trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map = device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=False)
    model.to(device)
    try: 
        print("------------This script is used on model llama-3-8b---------------")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load data and generate predictions
        test_data_path = '/scratch/project_2008167/thesis/data/test_data_llama3_no_outliers.json'
        results_path = '/scratch/project_2008167/thesis/evaluation/llama3_8b_automatic_evaluation_final.json'
        
        formatted_data = load_test_data(test_data_path)
        generate_and_save_predictions(model, tokenizer, formatted_data, results_path, batch_size=4)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
