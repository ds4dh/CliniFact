import json

def filter_jsonl(input_file, output_file, filter_key, filter_value):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            if record.get(filter_key) == filter_value:
                json.dump(record, outfile)
                outfile.write('\n')

# Parameters
input_file = '/data/processed/c_stance_label_extracted.jsonl'
output_file = '/data/processed/c_stance_label_extracted_primary.jsonl'
filter_key = 'outcome_type'
filter_value = 'PRIMARY'

# Execute the filter function
filter_jsonl(input_file, output_file, filter_key, filter_value)

