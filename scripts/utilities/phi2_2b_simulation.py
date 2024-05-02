import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nessa01macias/phi-2_sustainability-qa", trust_remote_code=False, torch_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=False)

inputs = tokenizer('What are SDGs?', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=512)

text = tokenizer.batch_decode(outputs)[0]

print(text)
