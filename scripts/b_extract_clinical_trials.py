import json
import os
import nltk
import logging
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import jsonlines

# Ensure that required NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load BioBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def map_intervention_types(arms_interventions_module):
    """
    Maps intervention types to their labels.
    """
    intervention_type_mapping = {}
    for intervention in arms_interventions_module.get("interventions", []):
        for label in intervention.get("armGroupLabels", []):
            intervention_type_mapping[label] = intervention["type"]
    return intervention_type_mapping

def get_biobert_embeddings(texts):
    """
    Gets BioBERT embeddings for a list of texts using GPU if available.
    """
    # Check if CUDA (GPU support) is available and use it; otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Send model to the appropriate device
    model.to(device)

    # Prepare inputs and send them to the same device
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # Perform the model forward pass
        outputs = model(**inputs)
    
    # Calculate mean of embeddings, ensuring to move the result back to CPU if needed for further non-GPU processing
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
    
    return embeddings

def create_outcomeMeasures_groups_to_armGroups_mapping(outcomeMeasures_groups_entries, armGroups_entries):
    """
    Creates a mapping based on cosine similarity between outcome measures and arm groups using BioBERT embeddings.
    """
    outcomeMeasures_groups_titles = [entry['title'] for entry in outcomeMeasures_groups_entries]
    armGroups_labels = [entry['label'] for entry in armGroups_entries]

    mapping = {(entry['title'], entry.get('description', '')): [] for entry in outcomeMeasures_groups_entries}

    embeddings_outcomeMeasures_groups = get_biobert_embeddings(outcomeMeasures_groups_titles)
    embeddings_armGroups = get_biobert_embeddings(armGroups_labels)
    cos_similarities = cosine_similarity(embeddings_outcomeMeasures_groups, embeddings_armGroups)

    for i, outcomeMeasure_group in enumerate(outcomeMeasures_groups_entries):
        max_sim_index = cos_similarities[i].argmax()
        most_similar_armGroup_entry = armGroups_entries[max_sim_index]
        if most_similar_armGroup_entry.get('label'):
            mapping[(outcomeMeasure_group['title'], outcomeMeasure_group.get('description', ''))].append({
                'label': most_similar_armGroup_entry['label'],
                'type': most_similar_armGroup_entry.get('type', '')
            })

    return mapping

def extract_information(json_data):
    """
    Extracts relevant information from JSON data and formats it.
    """
    try:
        nct_id = json_data['protocolSection']['identificationModule']['nctId']
        outcome_measures = json_data['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']
        armGroups_entries = json_data['protocolSection']['armsInterventionsModule'].get('armGroups', [])

        # Only process if armGroups are present
        if len(armGroups_entries) != 2:
            return []
        
        outcome_info = []
        for measure in outcome_measures:
            p_values = []
            stats_methods = []
            ni_types = []
            analyses = measure.get('analyses', [])
            for analysis in analyses:
                p_value = analysis.get('pValue')
                stats_method = analysis.get('statisticalMethod')
                ni_type = analysis.get('nonInferiorityType')
    
                if p_value is not None:
                    p_values.append(p_value)
                    stats_methods.append(stats_method)
                    ni_types.append(ni_type)

                    detailed_entries = measure.get('groups', [])

                    mapping = create_outcomeMeasures_groups_to_armGroups_mapping(detailed_entries, armGroups_entries)
                    intervention_type_mapping = map_intervention_types(json_data['protocolSection']['armsInterventionsModule'])

                    group_1 = mapping[(detailed_entries[0]['title'], detailed_entries[0].get('description', ''))][0]
                    group_2 = mapping[(detailed_entries[1]['title'], detailed_entries[1].get('description', ''))][0]

                    measure_info = {
                        'nctId': nct_id,
                        'outcome_type': measure['type'], 
                        'outcome_title': measure['title'], 
                        'outcome_description': measure.get('description', 'No Description Available'),
                        'outcome_timeFrame':  measure['timeFrame'],                
                        'pValues': p_values,
                        'statisticalMethods': stats_methods,
                        'nonInferiorityTypes': ni_types,
                        'group1_title': detailed_entries[0]['title'],
                        'group1_description': detailed_entries[0].get('description', ''),
                        'group1_arm_group_type': group_1['type'],
                        'group1_intervention_label': intervention_type_mapping.get(group_1['label'], ''),
                        'group2_title': detailed_entries[1]['title'],
                        'group2_description': detailed_entries[1].get('description', ''),
                        'group2_arm_group_type': group_2['type'],
                        'group2_intervention_label': intervention_type_mapping.get(group_2['label'], '')
                    }
                    outcome_info.append(measure_info)
        
        return outcome_info
    except KeyError as e:
        logging.error(f"KeyError: {e} in {json_data['protocolSection']['identificationModule']['nctId']}")
        return []

if __name__ == '__main__':
    directory_path = '/data/raw/ctg-studies'
    output_file = '/data/processed/b_clinical_trials_extracted.jsonl'

    try:
        with open(output_file, 'w') as outfile:
            for filename in os.listdir(directory_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(directory_path, filename)
                    try:
                        with open(file_path, 'r') as file:
                            json_data = json.load(file)
                            extracted_info = extract_information(json_data)
    
                            for info in extracted_info:
                                json.dump(info, outfile)
                                outfile.write('\n')
                    except json.JSONDecodeError as e:
                        logging.error(f"JSONDecodeError: {e} in file {file_path}")
                    except Exception as e:
                        logging.error(f"Error processing file {file_path}: {e}")
    
        logging.info(f"Extracted data saved to {output_file}")
    except IOError as e:
        logging.error(f"IOError: {e} while writing to {output_file}")