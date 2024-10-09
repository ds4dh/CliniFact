import json
import random
import os
from metapub import PubMedFetcher
import pandas as pd

def process_data_binary(json_line):
    """Process a JSON line for binary data."""
    data = json.loads(json_line)
    processed_data = {
        'nctId': data['nctId'],
        'outcome_type': data['outcome_type'],
        'outcome_title': data['outcome_title'],
        'outcome_timeFrame': data['outcome_timeFrame'],
        'intervention_group_title': data['intervention_group_title'],
        'comparator_group_title': data['comparator_group_title'],
        'PMID': data['resultPMIDs'][0] if data['resultPMIDs'] else None, # [0] because we only included the samples with single paper as a result reference
        'label': data['label'],

        'outcome_description': data['outcome_description'],
        'pValues': data['pValues'],
        'statisticalMethods': data['statisticalMethods'],
        'nonInferiorityTypes': data['nonInferiorityTypes'],

        'intervention_group_description': data['intervention_group_description'],
        'intervention_group_intervention_label': data['intervention_group_intervention_label'],
        'intervention_group_arm_group_type': data['intervention_group_arm_group_type'],

        'comparator_group_description': data['comparator_group_description'],
        'comparator_group_intervention_label': data['comparator_group_intervention_label'],
        'comparator_group_arm_group_type': data['comparator_group_arm_group_type']

    }
    return processed_data

def process_data_background(json_line):
    """Process a JSON line for background data."""
    data = json.loads(json_line)
    base_data = {
        'nctId': data['nctId'],
        'outcome_type': data['outcome_type'],
        'outcome_title': data['outcome_title'],
        'outcome_timeFrame': data['outcome_timeFrame'],
        'intervention_group_title': data['intervention_group_title'],
        'comparator_group_title': data['comparator_group_title'],
        'label': data['label'],

        'outcome_description': data['outcome_description'],
        'pValues': data['pValues'],
        'statisticalMethods': data['statisticalMethods'],
        'nonInferiorityTypes': data['nonInferiorityTypes'],

        'intervention_group_description': data['intervention_group_description'],
        'intervention_group_intervention_label': data['intervention_group_intervention_label'],
        'intervention_group_arm_group_type': data['intervention_group_arm_group_type'],

        'comparator_group_description': data['comparator_group_description'],
        'comparator_group_intervention_label': data['comparator_group_intervention_label'],
        'comparator_group_arm_group_type': data['comparator_group_arm_group_type']        
    }

    processed_data = []
    if data.get('backgroundPMIDs'):
        for pmid in data['backgroundPMIDs']:
            new_entry = base_data.copy()
            new_entry['PMID'] = pmid
            processed_data.append(new_entry)
    else:
        processed_data.append(base_data)

    return processed_data

def remove_duplicates_jsonl(input_file, output_file):
    unique_entries = set()
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            # Convert JSON object to a string to use it as a set element
            json_str = json.dumps(json_obj, sort_keys=True)
            if json_str not in unique_entries:
                unique_entries.add(json_str)
                outfile.write(line)

if __name__ == "__main__":
    
    # File paths for input and output files
    file_path_stance_labels_pmids_validate_refute = '/data/processed/e_stance_primary_evidence.jsonl'
    file_path_dataset_stance_binary = '/data/processed/f_primary_binary.jsonl'
    file_path_stance_labels_abstracts_background = '/data/processed/e_stance_primary_background.jsonl'
    file_path_dataset_stance_no_info = '/data/processed/f_primary_no_info.jsonl'

    # Process binary stance data
    with open(file_path_stance_labels_pmids_validate_refute, 'r') as file, open(file_path_dataset_stance_binary, 'w') as outfile:
        for line in file:
            processed_data = process_data_binary(line)
            json.dump(processed_data, outfile)
            outfile.write('\n')

    # Process background stance data
    with open(file_path_stance_labels_abstracts_background, 'r') as file, open(file_path_dataset_stance_no_info, 'w') as outfile:
        for line in file:
            entries = process_data_background(line)
            for entry in entries:
                json.dump(entry, outfile)
                outfile.write('\n')

    remove_duplicates_jsonl('/data/processed/f_primary_binary.jsonl', '/data/processed/f_primary_binary_1.jsonl')
    remove_duplicates_jsonl('/data/processed/f_primary_no_info.jsonl', '/data/processed/f_primary_no_info_1.jsonl')


