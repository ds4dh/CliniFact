import json

# Function to interpret a p-value string and return a tuple (value, relation)
def interpret_p_value(p_value_str):
    # Remove spaces and convert to lower case
    p_value_str = p_value_str.replace(" ", "").lower()

    # Determine the relation and convert the rest to float
    if "<" in p_value_str:
        relation = "lt"
        numeric_part = p_value_str.split("<")[-1]
    elif ">" in p_value_str:
        relation = "gt"
        numeric_part = p_value_str.split(">")[-1]
    else:
        relation = "eq"
        numeric_part = p_value_str.replace("=", "")

    # Convert numeric part to float
    try:
        value = float(numeric_part)
    except ValueError:
        value = None

    return value, relation

def determine_label(data):
    for p_value_str in data['pValues']:
        value, relation = interpret_p_value(p_value_str)
        if value is not None and ((relation == "lt" and value <= 0.05) or (relation == "eq" and value <= 0.05)):
            return "positive"
    return "negative"

# File path to your JSONL file
original_file_path = '/data/processed/b1_clinical_trials_extracted_intervention_comparator.jsonl'
new_file_path = '/data/processed/c_stance_label_extracted.jsonl'

with open(original_file_path, 'r') as infile, open(new_file_path, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        data['stance_label'] = determine_label(data)
        json.dump(data, outfile)
        outfile.write('\n')

print(f"Updated data saved to {new_file_path}")