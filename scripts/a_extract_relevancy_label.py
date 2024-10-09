import os
import json
import csv
from Bio import Entrez
from dateutil import parser
import re

def walk_json_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == "__main__":

    directory = '/data/raw/ctg-studies'
    output_file = '/data/processed/a_relevancy_labels_inc_derived.csv'

    # Get list of JSON files
    file_list = walk_json_files(directory)

    # Extract relevancy labels
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['nctId', 'PMID', 'Reference_Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_path in file_list:

            # Extract trial id from file name
            trial_id = os.path.basename(file_path).split('.')[0]
            
            # Parse JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            
            # Extract references
            references = data.get('protocolSection', {}).get('referencesModule', {}).get('references', [])
            results_date_str = data.get('protocolSection', {}).get('statusModule', {}).get('resultsFirstSubmitDate', '')

            # Parse results date using dateutil.parser
            results_date = parser.parse(results_date_str) if results_date_str else None

            # Extract paper id and label for each reference
            for reference in references:

                # Extract PMID
                pmid = reference.get('pmid', 'N/A')
                # Extract reference type
                reference_type = reference.get('type', 'N/A')
                
                label = None
                if reference_type == 'RESULT':
                    label = 1
                elif reference_type == 'BACKGROUND':
                    label = 0
                elif reference_type == 'DERIVED':
                    label = 2

                # Write data to CSV only if pmid and reference_type are not 'N/A'
                if pmid != 'N/A' and label is not None:
                    writer.writerow({'nctId': trial_id, 'PMID': pmid, 'Reference_Label': label})