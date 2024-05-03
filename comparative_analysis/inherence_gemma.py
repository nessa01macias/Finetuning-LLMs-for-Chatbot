import os
import json
from transformers import AutoTokenizer, pipeline
from datasets import Dataset
import pandas as pd

# Function to load data from a JSON file where each line is a separate JSON object
def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        data = [json.loads(line) for line in file if line.strip()]
    return data

# Define the path to your data file
cwd = os.getcwd()
data_path = os.path.join(cwd, 'table_prompts.json')
test_data = load_data_from_json(data_path)
test_df = pd.DataFrame(test_data)
test_dataset = Dataset.from_pandas(test_df)

# Create an iterator that yields prompts from the dataset
def prompt_iterator(dataset):
    for item in dataset:
        # This assumes 'question' is the key in your dataset that contains the text to generate from
        yield item['question']

# Setup the model and tokenizer
model = "nessa01macias/phi-2_sustainability-qa"
tokenizer = AutoTokenizer.from_pretrained(model)

# Create the pipeline for text generation
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # torch_dtype=torch.float16,
    device_map="auto"  # Automatically use the GPU if available
    # batch_size=2  # Adjust the batch size according to your GPU's capability
)

# Process the prompts using the pipeline
for output in text_generation_pipeline(prompt_iterator(test_dataset), max_new_tokens=100):
    for response in output:  # Iterate through the list of responses
        print(f"Result: {response['generated_text']}")  # Corrected to access each dictionary in the list

print("these were the rsults", output)
