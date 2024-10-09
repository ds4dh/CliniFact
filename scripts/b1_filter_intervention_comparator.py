

import json

def process_jsonlines(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            record = json.loads(line)

            # Streamlined filtering 
            if (record.get('group1_arm_group_type') == 'EXPERIMENTAL' and 
                record.get('group2_arm_group_type') != 'EXPERIMENTAL') or \
               (record.get('group2_arm_group_type') == 'EXPERIMENTAL' and 
                record.get('group1_arm_group_type') != 'EXPERIMENTAL'):

                # Determine intervention and comparator groups
                if record['group1_arm_group_type'] == 'EXPERIMENTAL':
                    intervention_group = 'group1'
                    comparator_group = 'group2'
                else:
                    intervention_group = 'group2'
                    comparator_group = 'group1'

                # Create new record, starting with all original fields
                new_record = record.copy()

                # Add new fields and remove old fields
                new_record['intervention_group_title'] = new_record.pop(f'{intervention_group}_title')
                new_record['intervention_group_description'] = new_record.pop(f'{intervention_group}_description')
                new_record['intervention_group_intervention_label'] = new_record.pop(f'{intervention_group}_intervention_label')
                new_record['intervention_group_arm_group_type'] = new_record.pop(f'{intervention_group}_arm_group_type')

                new_record['comparator_group_title'] = new_record.pop(f'{comparator_group}_title')
                new_record['comparator_group_description'] = new_record.pop(f'{comparator_group}_description')
                new_record['comparator_group_intervention_label'] = new_record.pop(f'{comparator_group}_intervention_label')
                new_record['comparator_group_arm_group_type'] = new_record.pop(f'{comparator_group}_arm_group_type')

                # Write to JSON Lines directly
                f_out.write(json.dumps(new_record) + '\n')

# Usage (same as before)
input_file = '/data/processed/b_clinical_trials_extracted.jsonl'  # Adjust file path
output_file = '/data/processed/b1_clinical_trials_extracted_intervention_comparator.jsonl'  # .jsonl extension
process_jsonlines(input_file, output_file)