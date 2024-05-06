import json

def format_prompt(instruction, input_text=""):
    """Format the prompt based on the provided instruction and optional input."""
    return f"{instruction}\n{input_text}"

def convert_data(file_path, output_path):
    """Convert data from current format to the expected metric computation format."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]

    formatted_data = []
    for index, item in enumerate(data):
        if 'prompt' in item and 'generated_answer' in item and 'reference_answer' in item:
            # Extracting and splitting the prompt part
            prompt_sections = item['prompt'].split("### Response:\n")
            if len(prompt_sections) > 0:
                prompt_section = prompt_sections[0].strip()
                instruction_parts = prompt_section.split("### Instruction:\n")
                input_parts = prompt_section.split("### Input:\n")
                
                instruction = instruction_parts[1].split("\n\n### Input:\n")[0].strip() if len(instruction_parts) > 1 else ""
                input_text = input_parts[1].strip() if len(input_parts) > 1 else ""
                
                # Remove specific phrases
                instruction = instruction.replace("### Instruction:", "").strip()
                input_text = input_text.replace("### Input:", "").strip()
                
                generated_answer = item['generated_answer'].replace(prompt_section, "").strip()
                reference_answer = item['reference_answer'].strip()

                formatted_entry = {
                    'index': index,
                    'question': format_prompt(instruction, input_text),
                    'generated_answer': generated_answer,
                    'reference_answer': reference_answer
                }
                formatted_data.append(formatted_entry)

    # Save the formatted data
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Specify the path to your specific JSON file and where you want the formatted output to go
file_path = '../results/finetuned/llama3_8b_automatic_evaluation_final.json'
output_path = '../results/finetuned/llama3_8b_automatic_evaluation_cleaned.json'

# Process the file
convert_data(file_path, output_path)
