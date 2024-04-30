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



def save_to_json(output_path, data):
    """ Saves or appends data to a JSON file """
    with open(output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def main():

    # Load English tokenizer, tagger, parser, and NER
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000 # or higher

    
    # Specify the directory where the PDF files are stored
    pdf_dir = os.path.join(os.getcwd(), 'sustainability_pdfs')
    files_names = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"2. Found {len(files_names)} PDF files in {pdf_dir}.")
    results_dir = os.path.join(pdf_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Dictionary to hold all Q&A pairs for each file
    total_tokens = []
    for filename in files_names:
        print(f"3. Reading file: {filename}")
        tokens_number = 0


        file_path = os.path.join(pdf_dir, filename)
        total_tokens.append({file_path: tokens_number})


    save_to_json(os.path.join(results_dir, 'tokens_count.json'), total_tokens)
    print(f"8. All Q&A pairs saved to {results_dir}")

if __name__ == "__main__":
    main()