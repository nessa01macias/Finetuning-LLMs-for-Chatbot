import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nessa01macias/phi-2_sustainability-qa", trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False, pad_token_id=tokenizer.eos_token_id)

propmt = "Ian had twenty roses. He gave six roses to his mother, nine roses to his grandmother, four roses to his sister, and he kept the rest. How many roses did Ian keep? Give me reasons, before answering the question"

inputs = tokenizer(propmt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_new_tokens=50)

text = tokenizer.batch_decode(outputs)[0]c

print(text)
