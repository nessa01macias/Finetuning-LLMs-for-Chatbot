from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(model, "nessa01macias/phi1.5_finetuned_sustainability_qa", from_transformers=True)
model = peft_model.merge_and_unload()
print(model)
model.push_to_hub("nessa01macias/phi1.5_finetuned_sustainability_qa")
print("saved to hugging face")