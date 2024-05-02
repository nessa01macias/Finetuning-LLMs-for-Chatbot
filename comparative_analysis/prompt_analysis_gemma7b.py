import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_questions(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = [item.get('question', item.get('text', '')) for item in data['reference_answer'] if 'question' in item or 'text' in item]
    return questions

def prompt_model(model, tokenizer, prompt: str):
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False, padding=True, truncation=True, max_length=700)
    outputs = model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()
    inference_time = end_time - start_time
    text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return text, inference_time

def main():

    # Set working directory and list files
    cwd = os.getcwd()
    questions_path = os.path.join(cwd, 'comparative_evaluation.json')

    questions = load_questions(questions_path)
    print(questions)

    # Models and tokenizers
    models_and_tokenizers = {
        # "phi2_finetuned": (AutoModelForCausalLM.from_pretrained("nessa01macias/phi-2_sustainability-qa", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = 'auto'),
        #                    AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False)),
        # "phi2": (AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=False),
        #          AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False)),
        # "gemma_2b_finetuned": (AutoModelForCausalLM.from_pretrained("nessa01macias/gemma-2b_sustainability-qa", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = 'auto'),
        #                        AutoTokenizer.from_pretrained("google/gemma-2b", trust_remote_code=False)),
        # "gemma_2b": (AutoModelForCausalLM.from_pretrained("google/gemma-2b", trust_remote_code=False),
        #                 AutoTokenizer.from_pretrained("google/gemma-2b", trust_remote_code=False)),
        "gemma_7b_finetuned": (AutoModelForCausalLM.from_pretrained("nessa01macias/gemma-7b_sustainability-qa", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = 'auto'),
                               AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=False)),
        "gemma_7b": (AutoModelForCausalLM.from_pretrained("google/gemma-7b", trust_remote_code=False, low_cpu_mem_usage=True, device_map='auto'),
                        AutoTokenizer.from_pretrained("google/gemma-7b", trust_remote_code=False)),
        # "llama3_8b_finetuned": (AutoModelForCausalLM.from_pretrained("nessa01macias/llama-3-8b_sustainability-qa", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = 'auto'),
        #                        AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=False)),
        # "llama3_8b": (AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", low_cpu_mem_usage=True, trust_remote_code=False, torch_dtype=torch.float16, device_map = 'auto'),
        #                 AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=False))
    }

    responses = []

    for model_name, (model, tokenizer) in models_and_tokenizers.items():
        for prompt_id, prompt_text in enumerate(questions, start=1):
            response, inference_time = prompt_model(model, tokenizer, prompt_text)
            responses.append({
                "prompt_id": prompt_id,
                "question": prompt_text,
                "model": model_name,
                "answer": response,
                "inference_time": inference_time
            })

    with open('model_responses_gemma.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()



