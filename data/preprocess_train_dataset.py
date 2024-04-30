import json
import os

pdf_dir = os.getcwd()  # Adjust this if necessary to point to the correct directory
combined_data_path = os.path.join(pdf_dir, 'final', 'train_data_with_prompts_OLD.json')

# Function to remove the specified prefix from the text
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

# Function to replace ".\nAnswer: \nAnswer" with "\nAnswer "
def replace_answer_text(text):
    return text.replace(".\nAnswer: \nAnswer", " \nAnswer")

# Specify the prefix to be removed
prefix_to_remove = "Question: You are an AI assistant. You need to make a concise summary from the following text: "

# Prepare to write the modified data to a new file
output_path = os.path.join(pdf_dir, 'final', 'output_file.json')
with open(output_path, 'w', encoding='utf-8') as output_file:
    # Read the file line by line and process each line as a separate JSON object
    with open(combined_data_path, 'r', encoding='utf-8') as file:
        first_entry = True
        for line in file:
            try:
                entry = json.loads(line)
                if 'text' in entry:
                    entry['text'] = remove_prefix(entry['text'], prefix_to_remove)
                    entry['text'] = replace_answer_text(entry['text'])
                # Write each entry to a new line in the output file
                if not first_entry:
                    output_file.write('\n')
                output_file.write(json.dumps(entry, ensure_ascii=False))
                first_entry = False
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

print(f"Processing complete. The modified data has been saved to '{output_path}'.")
