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
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set working directory and list files
cwd = os.getcwd()
train_data = os.path.join(cwd, 'train_data')
files = os.listdir(train_data)
print(f"Files in '{train_data}': {files}")

# Filter and load JSON data files
split_files = [f for f in files if f.startswith('train_data_split_') and f.endswith('.json')]
data_cleaned = []
errors = []

for file_name in split_files:
    file_path = os.path.join(train_data, file_name)  # Corrected to use 'train_data'
    with open(file_path, 'r', encoding='utf8') as file:
        try:
            data = json.load(file)
            data_cleaned.extend({'text': item['text']} for item in data if 'text' in item)
        except json.JSONDecodeError as e:
            errors.append((file_name, str(e)))

# Load data into a pandas DataFrame and report any errors
df = pd.DataFrame(data_cleaned)
if errors:
    print(f"Errors encountered: {len(errors)}")
    for error in errors[:10]:
        print(error)

print(f"Data loaded: {len(df)}")
print(df.head())

# Convert DataFrame to Hugging Face dataset
data = Dataset.from_pandas(df)

# Set model details
base_model = "microsoft/phi-2"
new_model = "phi-2_sustainability-qa"

# Initialize tokenizer and model with BitsAndBytes quantization
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"

print("Tokenizer loaded")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
print("Config loaded")

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": 0},
    revision="refs/pr/23" #the main version of Phi-2 doesn’t support gradient checkpointing (while training this model)
)

print("Model loaded")

model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Prepare model for training with gradient checkpointing
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Set training arguments
training_arguments = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 1,
    fp16 = True,
    bf16 = False,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 4,
    gradient_checkpointing = True,
    max_grad_norm = 0.3,
    learning_rate = 2e-4,
    weight_decay = 0.001,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    max_steps = -1,
    warmup_ratio = 0.03,
    group_by_length = True,
    save_steps = 0,
    logging_steps = 25,
)
print("Training arguments set")


# Configure and start the trainer
peft_config = LoraConfig(
    r= 4,
    lora_alpha= 16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["Wqkv", "fc1", "fc2"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length= 40,
    tokenizer=tokenizer,
    args=training_arguments
)

print("Trainer configured with LoraConfig")


# Train and save model
trainer.train()
trainer.model.save_pretrained(new_model)

print("Model trained saved")

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