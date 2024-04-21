import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model

def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer


model = load_quantized_model(model_name)
tokenizer = initialize_tokenizer(model_name)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

context = """Progress toward the United Nations Sustainable Development Goals (SDGs) has
been hindered by a lack of data on key environmental and socioeconomic indicators,
which historically have come from ground surveys with sparse temporal and
spatial coverage. Recent advances in machine learning have made it possible to
utilize abundant, frequently-updated, and globally available data, such as from
satellites or social media, to provide insights into progress toward SDGs. """

prompt = f"[INST] Please read the following context and generate a question and answer pair that captures the most important information. Provide the question first, followed by the answer, and ensure the format is clean with no additional text or symbols. Context: {context} [/INST]"

sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)

print(sequences[0]['generated_text'])


