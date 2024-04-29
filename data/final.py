import json
import os
from PyPDF2 import PdfReader
from cleantext import clean
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import spacy
import re
import torch

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = []
        keywords = ["acknowledgments", "references", "citations", "bibliography", "annexes", "end notes"]
        intro_sections = ["contents", "table of contents"]  # Sections that should not trigger text cut-off
        capture_text = True  # This flag determines whether to continue capturing text

        for page in reader.pages:
            if not capture_text:
                break  # Exit the loop if we've decided to stop capturing text

            page_text = page.extract_text()
            if page_text:
                page_text_lower = page_text.lower()  # Use lower case for matching

                # Check for introductory sections that should override keyword checks
                if any(intro in page_text_lower for intro in intro_sections):
                    text.append(page_text)  # Skip keyword checks and append the current page text
                    continue

                # Check for keywords followed by a newline or colon
                for keyword in keywords:
                    # Look for keywords followed by ':' or a newline in the original case text
                    patterns = [keyword + ":\n", keyword + "\n"]
                    found = False
                    for pattern in patterns:
                        if pattern in page_text_lower:
                            # Cut off the text at the start of the keyword pattern
                            keyword_index = page_text_lower.find(pattern)
                            page_text = page_text[:keyword_index]
                            capture_text = False  # Set flag to false to stop capturing text after this page
                            found = True
                            break
                    if found:
                        break
                if capture_text:
                    text.append(page_text)  # Append the page text if not cut off
        return "\n".join(text).replace('-\n', '')  # Handle hyphenated line breaks
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def clean_text(text, nlp) -> str:
    # First, clean the text with the existing cleaning parameters
    cleaned_text = clean(text,
                 fix_unicode=True,
                 to_ascii=True,
                 lower=False,
                 no_line_breaks=False,
                 no_urls=True,
                 no_emails=True,
                 no_phone_numbers=True,
                 no_numbers=False,
                 no_digits=False,
                 no_currency_symbols=True,
                 no_punct=False,
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_phone_number="",
                 replace_with_number="",
                 replace_with_digit="",
                 replace_with_currency_symbol="",
                 lang="en")
    
    # Now apply SpaCy NER to remove names and organizations
    doc = nlp(cleaned_text)
    entities_to_remove = ['PERSON', 'ORG']  # Define the entity types to remove
    intervals_to_remove = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in entities_to_remove]
    
    # Build the final cleaned text by excluding entities marked for removal
    final_cleaned_text = ""
    start_idx = 0
    for start, end in intervals_to_remove:
        final_cleaned_text += cleaned_text[start_idx:start]  # Add text up to the start of the entity
        start_idx = end  # Move the start index to the end of the current entity
    final_cleaned_text += cleaned_text[start_idx:]  # Add any remaining text after the last entity
    return final_cleaned_text


def get_chunked_text(text, max_length=500):
    """
    Split the text into chunks without cutting sentences in the middle.
    Specifically handles not splitting at periods within parentheses, brackets, or braces.
    """
    chunks = []
    current_chunk = ""
    open_brackets = {'(': 0, '[': 0, '{': 0}
    close_brackets = {')': '(', ']': '[', '}': '{'}

    i = 0
    while i < len(text):
        char = text[i]
        if char in open_brackets:
            open_brackets[char] += 1
        elif char in close_brackets:
            if open_brackets[close_brackets[char]] > 0:
                open_brackets[close_brackets[char]] -= 1
        
        current_chunk += char

        # Check if current character is a period and it's safe to split
        if char == '.' and i + 1 < len(text) and text[i + 1] == ' ' and all(v == 0 for v in open_brackets.values()):
            # Check if adding the next sentence would exceed max_length
            next_space = text.find(' ', i + 1)
            if next_space == -1:  # Handle end of text without spaces
                next_space = len(text)
            if len(current_chunk) + (next_space - i) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            i = next_space  # Move to the next space
        else:
            i += 1

    # Append the last chunk if any
    if current_chunk and len(current_chunk) > 100:
        chunks.append(current_chunk.strip())

    return chunks

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

def initialize_pipeline(model, tokenizer):
    generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            batch_size=6,
            device_map="auto",
            truncation=True,
            repetition_penalty=1.1 
        )
    
    return generator 


def generate_qa_pairs(text, generator):
    prompt = f"[INST] Please read the following context and generate a question and answer pair that captures the most important information. Make the question and answer detailed and with context. Provide the question first, followed by the answer, and ensure the format is clean with no additional text or symbols. Context: {text} [/INST]"
    try:
        sequences = generator(
            prompt,
            do_sample=True,
            max_new_tokens=1000, 
            min_new_tokens= 200,
            temperature=0.2, 
            top_k=50, 
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
            num_return_sequences=1
        )
    finally:
    # Clear GPU memory cache to prevent fragmentation and OOM errors
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

    return sequences[0]['generated_text']

def parse_qa_from_text(text):
    """
    Parses multiple question and answer pairs from the generated text using regular expressions
    to handle different cases and formatting variations.
    """
    try:
        qa_pairs = []
        # Regular expression to find Q&A patterns
        pattern = re.compile(r'\n*Question:\s*(.*?)\s*\n+Answer:\s*(.*?)\s*(?=\n*Question:|$)', re.IGNORECASE)
        
        matches = pattern.finditer(text)
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer:  # Ensure non-empty entries
                qa_pairs.append({'question': question, 'answer': answer})
        
        return qa_pairs
    except Exception as e:
        print(f"Error parsing Q&A: {e}")
        return []


def save_to_json(output_path, data):
    """ Saves or appends data to a JSON file """
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_pdf_file(file_path, nlp, generator):
    """ Process a single PDF file to extract, clean, and generate Q&A pairs """
    print(f"3. Processing file: {file_path}")
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        return []
        
    cleaned_text = clean_text(raw_text, nlp)
    print(f"4. Cleaned text from file {file_path}.")

    chunks = get_chunked_text(cleaned_text)
    print(f"5. Created chunks and now generating Q&A pairs from file {file_path}...")

    qa_pairs = []

    for chunk in chunks:
        qa_text = generate_qa_pairs(chunk, generator)
        parsed_qa = parse_qa_from_text(qa_text)
        qa_pairs.extend(parsed_qa)  # Extend list with new Q&A pairs
        print(parsed_qa)

    print(f"6. Q&A pairs generated for {file_path}")
    return qa_pairs


def main():

    # Load English tokenizer, tagger, parser, and NER
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000 # or higher

    # Load the quantized model and tokenizer
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    model = load_quantized_model(model_name)
    tokenizer = initialize_tokenizer(model_name)
    generator = initialize_pipeline(model, tokenizer)
    print("1. Model loaded successfully.")
    
    # Specify the directory where the PDF files are stored
    pdf_dir = os.path.join(os.getcwd(), 'sustainability_pdfs')
    files_names = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"2. Found {len(files_names)} PDF files in {pdf_dir}.")
    results_dir = os.path.join(pdf_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Dictionary to hold all Q&A pairs for each file
    all_qa_pairs = {}
    for filename in files_names:
        print(f"3. Reading file: {filename}")

        file_path = os.path.join(pdf_dir, filename)
        qa_pairs = process_pdf_file(file_path, nlp, generator)
        all_qa_pairs[filename] = qa_pairs
        save_to_json(os.path.join(results_dir, filename + '_results.json'), {filename: qa_pairs})
        print(f"7. Q&A pairs saved to {results_dir} for {filename}")

    save_to_json(os.path.join(results_dir, 'all_qa_dataset.json'), all_qa_pairs)
    print(f"8. All Q&A pairs saved to {results_dir}")

if __name__ == "__main__":
    main()