import json
import random
import os
import pandas as pd
from collections import Counter

def remove_null_rows(input_file_path):
    """Remove rows with null values or short abstracts."""
    null_count = 0
    short_abstract_count = 0
    label_counts = {'evidence': 0, 'inconclusive': 0, 'no_info': 0}
    data_list = {'evidence': [], 'inconclusive': [], 'no_info': []}

    with open(input_file_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            if any(value is None for value in data.values()):
                null_count += 1
                continue

            if word_count(data.get('article_abstract', '')) <= 15:
                short_abstract_count += 1
                continue

            label_stance = data.get('label')
            if label_stance in label_counts:
                label_counts[label_stance] += 1
                data_list[label_stance].append(data)

    return null_count, short_abstract_count, label_counts, data_list

def word_count(text):
    """Count the number of words in a text."""
    return len(text.split())

def calculate_stats(file_path):
    """Calculate and return word count statistics for specific fields."""
    index, outcome_title_counts, timeFrame_counts, intervention_counts, article_title_counts, article_abstract_counts = [], [], [], [], [], []

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            index.append(data.get('index', ''))
            outcome_title_counts.append(word_count(data.get('outcome_title', '')))
            timeFrame_counts.append(word_count(data.get('outcome_timeFrame', '')))
            intervention_counts.append(word_count(data.get('intervention', '')))  # Populate this list
            article_title_counts.append(word_count(data.get('article_title', '')))
            article_abstract_counts.append(word_count(data.get('article_abstract', '')))

    stats = {}
    for field, counts in zip(
        ['outcome_title', 'outcome_timeFrame', 'article_title', 'article_abstract'],
        [outcome_title_counts, timeFrame_counts, article_title_counts, article_abstract_counts]
    ):
        if counts:  # Check if the list is not empty
            min_count = min(counts)
            min_indices = [idx for idx, count in zip(index, counts) if count == min_count]
            stats[field] = {
                'mean': sum(counts) / len(counts),
                'min': min_count,
                'min_indices': min_indices,
                'max': max(counts)
            }
        else:
            stats[field] = {
                'mean': None,
                'min': None,
                'min_indices': [],
                'max': None
            }

    return stats

def downsample_no_info(data_list, target_count):
    """Downsample 'no_info' samples to match the target count."""
    no_info_samples = data_list['no_info']
    if len(no_info_samples) > target_count:
        random.seed(1234)
        data_list['no_info'] = random.sample(no_info_samples, target_count)

    return data_list

def interpret_p_value(p_value_str):
    """Interpret a p-value string and return a tuple (value, relation)."""
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

def update_non_inferiority_types(input_file_path, output_file_path):
 # The dictionary that maps machine-interpretable language to human-interpretable language
    mapping = {
        "SUPERIORITY": "Superiority",
        "NON_INFERIORITY": "Non-Inferiority",
        "EQUIVALENCE": "Equivalence",
        "OTHER": "Other",
        "NON_INFERIORITY_OR_EQUIVALENCE": "Non-Inferiority or Equivalence",
        "SUPERIORITY_OR_OTHER": "Superiority or Other",
        "NON_INFERIORITY_OR_EQUIVALENCE_LEGACY": "Non-Inferiority or Equivalence",
        "SUPERIORITY_OR_OTHER_LEGACY": "Superiority or Other"
    }
    
    updated_non_inferiority_counter = Counter()
    unique_combinations = Counter()

    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            
            if 'nonInferiorityTypes' in data and 'pValues' in data:
                non_inferiority_types = data['nonInferiorityTypes']
                p_values = data['pValues']
                
                # Update the nonInferiorityTypes field using the mapping
                updated_types = [mapping.get(item, item) for item in non_inferiority_types]
                
                # Determine if any p-value is less than or equal to 0.05
                has_significant_p_value = False
                for p_val_str in p_values:
                    value, relation = interpret_p_value(p_val_str)
                    if value is not None and ((relation == "lt" and value <= 0.05) or (relation == "eq" and value <= 0.05)):
                        has_significant_p_value = True
                        break

                if has_significant_p_value:
                    # Keep only types where corresponding p-value <= 0.05
                    valid_types = []
                    for type_, p_val_str in zip(updated_types, p_values):
                        value, relation = interpret_p_value(p_val_str)
                        if value is not None and ((relation == "lt" and value <= 0.05) or (relation == "eq" and value <= 0.05)):
                            valid_types.append(type_)
                    updated_types = valid_types

                    # Final check to collapse lists with identical elements to a single element
                    if len(set(updated_types)) == 1:
                        updated_types = [updated_types[0]]
                    else:
                        updated_types = list(set(updated_types))
                else:
                    # If no p-values <= 0.05, keep all unique elements
                    updated_types = list(set(updated_types))

                if 'Other' in updated_types and len(updated_types) == 1:
                    updated_types = []
                elif len(updated_types) >= 2:
                    if 'Other' in updated_types and 'Superiority' in updated_types and len(updated_types) == 2:
                        updated_types = ['Superiority or Other']
                    elif 'Non-Inferiority' in updated_types and 'Equivalence' in updated_types and len(updated_types) == 2:
                        updated_types = ['Non-Inferiority or Equivalence']
                    else:
                        updated_types = []

                data['nonInferiorityTypes'] = updated_types    
            
                if updated_types:
                    # Count the updated nonInferiorityTypes
                    updated_non_inferiority_counter.update(updated_types)
                    # Count unique combinations of updated nonInferiorityTypes
                    unique_combinations.update([tuple(updated_types)])
                    # Write the updated JSON object to the output file
                    outfile.write(json.dumps(data) + '\n')

        # Print statistics
        print("Updated nonInferiorityTypes counts:", dict(updated_non_inferiority_counter))
        print("Unique combinations of updated nonInferiorityTypes:", dict(unique_combinations))

        print("The 'nonInferiorityTypes' field has been analyzed and updated successfully for all records.")

if __name__ == "__main__":
    input_file_path = '/data/processed/g_primary_publication.jsonl'
    input_file_path_1 = '/data/processed/g_primary_publication-1.jsonl'
    output_file_path_1 = '/data/processed/g1_primary_publication_ready4split-1.jsonl'

    update_non_inferiority_types(input_file_path, input_file_path_1)

    # Remove null rows and short abstracts
    null_rows_count, short_abstract_rows_count, stance_label_counts, data_list = remove_null_rows(input_file_path_1)

    print(f"Number of rows containing null: {null_rows_count}")
    print(f"Number of rows containing very short abstracts: {short_abstract_rows_count}")
    print(f"Label counts before downsampling: {stance_label_counts}")

    # Calculate target count for 'no_info'
    target_no_info_count = stance_label_counts['evidence'] + stance_label_counts['inconclusive']

    # Downsample 'no_info' samples
    data_list = downsample_no_info(data_list, target_no_info_count)

    # Write the downsampled data to the output file
    with open(output_file_path_1, 'w') as outfile:
        for label in data_list:
            for data in data_list[label]:
                json.dump(data, outfile)
                outfile.write('\n')