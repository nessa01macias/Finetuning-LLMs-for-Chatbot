import os
from dataclasses import dataclass, field
from typing import optional
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
from tqdm.notebook import tqm
from trl import SFTTrainer
from huggingface_hub import intepreter_login