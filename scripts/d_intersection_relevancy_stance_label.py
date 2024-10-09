import pandas as pd
import json

# File paths
relevancy_labels_file = '/data/processed/a_relevancy_labels_inc_derived.csv'
stance_labels_file = '/data/processed/c_stance_label_extracted.jsonl'
output_file = '/data/processed/d_stance_and_relevancy_labels_mapped.jsonl'

# Read the relevancy labels using Pandas
df = pd.read_csv(relevancy_labels_file)

df_1 = df[df['Reference_Label'] == 1]
df_0 = df[df['Reference_Label'] == 0]

# Group by TrialID and create a list of PMIDs for each TrialID
trial_to_pmids_1 = df_1.groupby('nctId')['PMID'].apply(list).to_dict()
trial_to_pmids_0 = df_0.groupby('nctId')['PMID'].apply(list).to_dict()

# Read the stance labels JSONL file and add PMIDs to each entry
updated_stance_labels = []
with open(stance_labels_file, mode='r', encoding='utf-8') as file:
    for line in file:
        stance_data = json.loads(line)
        trial_id = stance_data['nctId']
        stance_data['resultPMIDs'] = trial_to_pmids_1.get(trial_id, [])
        stance_data['backgroundPMIDs'] = trial_to_pmids_0.get(trial_id, [])
        updated_stance_labels.append(stance_data)

# Save the updated stance labels to a new file (optional)
with open(output_file, 'w', encoding='utf-8') as file:
    for entry in updated_stance_labels:
        json.dump(entry, file)
        file.write('\n')