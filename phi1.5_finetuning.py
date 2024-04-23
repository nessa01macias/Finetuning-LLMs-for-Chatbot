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
combined_data = os.path.join(pdf_dir, 'data_generation', 'sustainability_pdfs', 'results', 'final', 'all_data_combined.json')

# Load JSON data
with open(combined_data, encoding="utf8") as file:
    data = json.load(file)

# Transform data into the desired format and create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame to verify contents
print(df.head())
print(df.shape)
