from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import transformers
import time
import pdb

start0=time.time()
model_dir = './mpt-7b-instruct'

config = AutoConfig.from_pretrained(
  model_dir,
  trust_remote_code=True,
  max_new_tokens=1024
)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
model.tie_weights()

model = load_checkpoint_and_dispatch(
    model, model_dir, device_map="auto", no_split_module_classes=["MPTBlock"]
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
#pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=500,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
diff0=time.time()-start0
print(diff0, "\n")

start1=time.time()
res=generate_text("Answer the following question:\nQ.My mother has recently been diagnosed with dementia, what support is available for her? \nA.")
print(res[0]["generated_text"])
diff1=time.time()-start1
print(diff1,"\n")

start2=time.time()
res=generate_text("Answer the following question:\nQ.Where can I share my story (about looking after someone with cognitive problems) and hear from others?  \nA.")
print(res[0]["generated_text"])
diff2=time.time()-start2
print(diff2)

pdb.set_trace()