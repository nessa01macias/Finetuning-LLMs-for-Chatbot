# Import necessary libraries
import os
import json
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# Environment configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set working directory and list files
cwd = os.getcwd()
train_data_path = os.path.join(cwd, 'train_data')
files = os.listdir(train_data_path)
print(f"Files in '{train_data_path}': {files}")

# Load and format the dataset
data_cleaned = []
errors = []
split_files = [f for f in files if f.startswith('train_data_split_') and f.endswith('.json')]

for file_name in split_files:
    file_path = os.path.join(train_data_path, file_name)
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            data = json.load(file)
            for item in data:
                if 'text' in item:
                    data_cleaned.append({'text': item['text']})
                else:
                    errors.append(f"Missing 'text' key in item from file {file_name}")
    except json.JSONDecodeError as e:
        errors.append((file_name, str(e)))

df = pd.DataFrame(data_cleaned)
if errors:
    print(f"Errors encountered: {len(errors)}")
    for error in errors[:10]:
        print(error)

print(df.head())
data = Dataset.from_pandas(df)

# Load model and tokenizer
base_model = "google/gemma-2b"
new_model = "gemma-2b_sustainability-qa"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Tokenizer loaded.")

# Configure BitsAndBytes for model quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
print("Config loaded.")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": 0}
)

print("Model loaded.")

model.config.use_cache = False
model.config.pretraining_tp = 1

# Training arguments
training_arguments = TrainingArguments(
    output_dir="./results_gemma",
    num_train_epochs=1,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 4,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type= "cosine"
)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=40,
    tokenizer=tokenizer,
    args=training_arguments,
)

print("Trainer configured with LoraConfig")

# Start training
trainer.train()
trainer.model.save_pretrained(new_model)

print("Training complete and model saved.")


# Reload model and merge it with LoRA parameters
print("Reloading model again to merge with LoRA parameters")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir="",
    device_map = 'auto'
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)