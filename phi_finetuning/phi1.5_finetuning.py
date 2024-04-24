from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
import pandas as pd
import torch
import json
import os

# Path setup for the JSON file containing the data
pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
combined_data = os.path.join(pdf_dir, 'data_generation', 'sustainability_pdfs', 'results', 'final', 'train_data.json')

# Load JSON data
data = []
with open(combined_data, 'r', encoding="utf8") as file:
    for line in file:
        data.append(json.loads(line))

# Transform data into the desired format and create a DataFrame
def process_entry(entry):
    parts = entry["text"].split("[/INST]")
    question = parts[0].replace("[INST]", "").strip()
    answer = parts[1].strip()
    return f"question: {question} answer: {answer}"

data_processed = [process_entry(entry) for entry in data]
df = pd.DataFrame(data_processed, columns=["text"])

# Display the DataFrame to verify contents
print(df.head(), df.shape)

# this is only for ttraining data
data = Dataset.from_pandas(df)

base_model = "microsoft/phi-1_5"
new_model = "phi-1_5-finetuned-sustainability-qa"

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define the configuration for the 4 BitsAndBytes quantization - Reducing the memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # saves an additional 0.4 bits per parameter, second quantization
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16 # only compatible with GPU - any GPU cuda > 11.2 should work
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"":0},
    trust_remote_code=True
)

print("basic model\n", model)

config = LoraConfig(
    r=16, #rank of the low-rank matrics
    lora_alpha=16, # escalating alfa/r 
    target_modules=["Wqkv", "out_proj"], # name of the layers to quantize, taken from the model architecture
    lora_dropout=0.05, # dropout rate to reduce overfitting
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config) # finetuning a small number of parameters while freezing the most part of the
# parameters during the training process 
model.print_trainable_parameters()

def tokenize(sample):
    tokenized_text =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
    return tokenized_text

tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

training_arguments = TrainingArguments( 
        output_dir="phi1.5_finetuned_sustainability_qa",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=100,
        optim = "paged_adamw_8bit",
        max_steps=1000, 
        num_train_epochs=1,
        push_to_hub=True
    )
     
trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.push_to_hub()
