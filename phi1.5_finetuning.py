import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments
)
from tqdm.notebook import tqdm
from trl import SFTTrainer
from huggingface_hub import interpreter_login
import os
import pandas as pd
import json

# Path setup for the JSON file containing the data
pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
sustainability_pdf = os.path.join(pdf_dir, 'sustainability_pdfs', 'results', 'final', 'final_results.json')

# Load JSON data
with open(sustainability_pdf, encoding="utf8") as file:
    data = json.load(file)

# Transform data into the desired format and create a DataFrame
formatted_data = [f"[INST] {item['question']} [/INST] {item['answer']}" for item in data]
df = pd.DataFrame(formatted_data, columns=['text'])

# Display the DataFrame to verify contents
print(df.head())
