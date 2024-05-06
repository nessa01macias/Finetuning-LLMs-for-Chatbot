import json

def clean_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for index, line in enumerate(data_lines):  # Using enumerate to keep an index
            if line.strip():
                data = json.loads(line)
                question = data['question'].replace('Question: ', '').strip()
                generated_answer = data['generated_answer']
                
                if generated_answer.startswith('Answer: '):
                    generated_answer = generated_answer.replace('Answer: ', '', 1)

                # Remove repeated prompt from the generated answer if it exists
                if question in generated_answer:
                    print("Example: ", question, generated_answer)
                    start_idx = generated_answer.find(question)
                    end_idx = start_idx + len(question)
                    generated_answer = generated_answer[:start_idx] + generated_answer[end_idx:]

                # Clean up additional whitespace and newlines
                generated_answer = ' '.join(generated_answer.split())

                # Skip entries with empty responses
                if not generated_answer.strip() or generated_answer == '':
                    continue
                
                data['question'] = question
                data['generated_answer'] = generated_answer
                data['index'] = index  # Preserve the original line index for tracking

                # Use json.dumps to properly handle Unicode characters
                json_string = json.dumps(data, ensure_ascii=False)
                outfile.write(json_string + '\n')

# Define paths
input_file = '../results/finetuned/llama3_8b_automatic_evaluation_final.json'
output_file = '../results/finetuned/llama3_8b_automatic_evaluation_cleaned.json'

clean_data(input_file, output_file)
