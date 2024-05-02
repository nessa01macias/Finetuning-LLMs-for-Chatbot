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


compute_dtype = torch.float16

model_name = "meta-llama/Meta-Llama-3-8B"
new_model = "llama-3-8b_sustainability-qa"


#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id
tokenizer.padding_side = 'left'


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        trust_remote_code=True,
        device_map={"": 0}
)

model = prepare_model_for_kbit_training(model)

#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

training_arguments = TrainingArguments(
        output_dir="./results_Llama3_8b",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=2,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
)

trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
)

print("Trainer configured with LoraConfig")

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

print("Training completed and model saved")


# Reload model and merge it with LoRA parameters
print("Reloading llama3-8b model again to merge with LoRA parameters")

#Load the base model with default precision
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype, device_map = 'auto')

#Load and activate the adapter on top of the base model
model = PeftModel.from_pretrained(model, new_model)

#Merge the adapter with the base model
model = model.merge_and_unload()
model.push_to_hub(new_model, use_temp_dir=False)
