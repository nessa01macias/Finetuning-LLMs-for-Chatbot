import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from cleantext import clean

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def clean_text(text):
    return clean(text,
                 fix_unicode=True,               # fix various unicode errors
                 to_ascii=True,                  # transliterate to closest ASCII representation
                 lower=False,                    # lowercase text
                 no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                 no_urls=True,                   # replace all URLs with a special token
                 no_emails=True,                 # replace all email addresses with a special token
                 no_phone_numbers=True,          # replace all phone numbers with a special token
                 no_numbers=False,               # replace all numbers with a special token
                 no_digits=False,                # replace all digits with a special token
                 no_currency_symbols=True,       # replace all currency symbols with a special token
                 no_punct=False,                 # fully remove punctuation
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="<DIGIT>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en")

def chunk_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def generate_qa_pairs(text_chunks, model, instruction="Generate a question-answer pair from the following text:"):
    qa_pairs = []
    for chunk in text_chunks:
        prompt = instruction + " " + chunk
        result = model(prompt) # THIS LINE WILL BE UPDATED WITH THE NEW FALCON 7B MODEL
        qa_pairs.append(result)
    return qa_pairs

def main():
    # Load the LLM from Hugging Face
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    # Extract and clean text from PDF
    pdf_dir = os.path.join(os.getcwd(), 'sustainability_pdfs')
    file_path = pdf_dir + '2111.04724v1.pdf'

    raw_text = extract_text_from_pdf(file_path)
    if raw_text:
        cleaned_text = clean_text(raw_text)

        # Chunk the text
        chunks = chunk_text(cleaned_text)

        # THIS WILL BE UPDATED WITH THE NEW FALCON 7B MODEL
        # Generate Q&A pairs with a custom instruction
        custom_instruction = "Please generate a question-answer pair from the following text:"
        qa_pairs = generate_qa_pairs(chunks, model, custom_instruction)
        #######################################################

        # Save results to a JSON file
        with open('qa_dataset.json', 'w') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
