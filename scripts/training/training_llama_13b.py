# Imports
import os
import json
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
    logging,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def formatting_prompts_func(example):
    output_texts = []
    for text in example['text']:
        # Split the text into the question and answer parts
        parts = text.split('\nAnswer: ')
        question = parts[0]  # This is everything before "Answer:"
        answer = parts[1] if len(parts) > 1 else ''  # Everything after "Answer:", if it exists
        
        # Now format it as required
        formatted_text = f"Input: {question}\n### Output: {answer}"
        output_texts.append(formatted_text)
    
    return {'text': output_texts}


# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Set working directory and list files
cwd = os.getcwd()
train_data_path = os.path.join(cwd, 'data', 'train_data.json')
val_data_path = os.path.join(cwd, 'data', 'validation_data.json')

# Load and format training data
with open(train_data_path, 'r', encoding='utf8') as file:
    train_data = [json.loads(line) for line in file if line.strip()]
train_df = pd.DataFrame(train_data)
train_dataset = Dataset.from_pandas(train_df)
print(f"Training data loaded: {len(train_df)} entries")

# Load and format validation data
with open(val_data_path, 'r', encoding='utf8') as file:
    val_data = [json.loads(line) for line in file if line.strip()]
val_df = pd.DataFrame(val_data)
val_dataset = Dataset.from_pandas(val_df)
print(f"Validation data loaded: {len(val_df)} entries")

# Set model details
base_model = "meta-llama/Llama-2-13b-hf"
new_model = "llama-2-13b_sustainability-qa"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded")

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
print("Config configuration set.")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": 0}
)
print("Model loaded.")

model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

training_arguments = TrainingArguments(
    output_dir="./results_llama",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)
print("Training arguments set.")

peft_config= LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length= 40,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func  
)

print("Trainer configured with LoraConfig")

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

print("Model trained saved")

# Reload model and merge it with LoRA parameters
print("Reloading llama model again to merge with LoRA parameters")
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