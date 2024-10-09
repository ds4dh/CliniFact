import json

def process_data_binary(json_line):
    """
    Process a single JSON line to add a 'label_stance' based on 'resultPMIDs' and 'label'.

    Args:
        json_line (str): A JSON string representing a single line from the file.

    Returns:
        dict or None: Modified data with 'label_stance' or None if conditions are not met.
    """
    # Parse the JSON line into a dictionary
    data = json.loads(json_line)

    # Check if the entry type is 'PRIMARY'
    if data['outcome_type'] == 'PRIMARY':
        # Check if there is exactly one 'resultPMID'
        if len(data.get('resultPMIDs', [])) == 1:
            # Assign 'validate' if the label is 'positive'
            if data['stance_label'] == 'positive':
                data['label'] = 'evidence'
                return data
            # Assign 'refute' if the label is 'negative'
            elif data['stance_label'] == 'negative':
                data['label'] = 'inconclusive'
                return data

    # Return None if conditions are not met
    return None

def process_data_background(json_line):
    """
    Process a single JSON line to add a 'label_stance' based on 'backgroundPMIDs'.

    Args:
        json_line (str): A JSON string representing a single line from the file.

    Returns:
        dict or None: Modified data with 'label_stance' or None if conditions are not met.
    """
    # Parse the JSON line into a dictionary
    data = json.loads(json_line)

    # Check if the entry type is 'PRIMARY'
    if data['outcome_type'] == 'PRIMARY':
        # Check if there is at least one 'backgroundPMID'
        if len(data.get('backgroundPMIDs', [])) >= 1:
            # Assign 'no_info' to 'label_stance'
            data['label'] = 'no_info'
            return data

    # Return None if conditions are not met
    return None

# Process the file to add 'label_stance' for 'validate' and 'refute'
with open('/data/processed/d_stance_and_relevancy_labels_mapped.jsonl', 'r') as file, open('/data/processed/e_stance_primary_evidence.jsonl', 'w') as outfile:
    # Read each line in the input file
    for line in file:
        # Process the line using the 'process_data_binary' function
        processed_data = process_data_binary(line)
        # Write the processed data to the output file if not None
        if processed_data is not None:
            json.dump(processed_data, outfile)
            outfile.write('\n')

# Process the file to add 'label_stance' for 'no_info'
with open('/data/processed/d_stance_and_relevancy_labels_mapped.jsonl', 'r') as file, open('/data/processed/e_stance_primary_background.jsonl', 'w') as outfile:
    # Read each line in the input file
    for line in file:
        # Process the line using the 'process_data_background' function
        processed_data = process_data_background(line)
        # Write the processed data to the output file if not None
        if processed_data is not None:
            json.dump(processed_data, outfile)
            outfile.write('\n')