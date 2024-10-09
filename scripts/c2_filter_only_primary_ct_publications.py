import json
import csv

# File paths
relevancy_labels_file = '/data/a_relevancy_labels_inc_derived.csv'
stance_labels_file = '/data/processed/c_stance_label_extracted_primary.jsonl'
output_file = '/data/processed/a1_relevancy_labels_inc_derived_filtered_primary.csv'

def extract_nct_ids_from_jsonl(jsonl_file):
    nct_ids = set()
    with open(jsonl_file, 'r') as infile:
        for line in infile:
            record = json.loads(line)
            nctId = record.get('nctId')
            if nctId:
                nct_ids.add(nctId)
    return nct_ids

def filter_csv_by_nct_ids(csv_file, output_file, nct_ids):
    with open(csv_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            if row['nctId'] in nct_ids:
                writer.writerow(row)

nctids = extract_nct_ids_from_jsonl(stance_labels_file)
filter_csv_by_nct_ids(relevancy_labels_file,output_file,nctids)
