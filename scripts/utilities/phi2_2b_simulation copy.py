import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False)

propmt = "What global goals were established to address the most pressing challenges of our time, and how do they build upon previous initiatives?"

inputs = tokenizer(propmt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_new_tokens=100)

text = tokenizer.batch_decode(outputs)[0]

print(text)
