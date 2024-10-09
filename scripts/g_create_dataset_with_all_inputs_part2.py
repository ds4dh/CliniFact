import json
import random
import os
from metapub import PubMedFetcher

def shuffle_and_index_files(file_path1, file_path2, output_file_path):
    """Shuffle and index data from two files, then save to an output file."""
    random.seed(1234)
    print(f"Random seed set to: {1234}")

    combined_data = []
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
        combined_data.extend(json.loads(line) for line in file1)
        combined_data.extend(json.loads(line) for line in file2)

    random.shuffle(combined_data)

    with open(output_file_path, 'w') as outfile:
        for index, data in enumerate(combined_data, start=1):
            data['index'] = index
            json.dump(data, outfile)
            outfile.write('\n')

def add_pubmed_abstracts(input_file_path, output_file_path):
    """Fetch and add PubMed abstracts to the dataset."""
    fetcher = PubMedFetcher()

    last_processed_index = 0
    try:
        with open(output_file_path, 'r') as outfile:
            for line in outfile:
                data = json.loads(line)
                if 'index' in data:
                    last_processed_index = max(last_processed_index, data['index'])
    except FileNotFoundError:
        pass

    with open(input_file_path, 'r') as file, open(output_file_path, 'a') as outfile:
        for line in file:
            data = json.loads(line)
            if data['index'] > last_processed_index:
                if 'PMID' in data:
                    try:
                        article = fetcher.article_by_pmid(str(data['PMID']))
                        if article:
                            data['article_title'] = article.title
                            data['article_abstract'] = article.abstract
                        else:
                            data['article_title'] = None
                            data['article_abstract'] = None
                    except InvalidPMID:
                        print(f"Invalid PMID: {data['PMID']}")
                        data['article_title'] = None
                        data['article_abstract'] = None
                else:
                    data['article_title'] = None
                    data['article_abstract'] = None

                json.dump(data, outfile)
                outfile.write('\n')

def update_labels(input_file_path, output_file_path):
    """Update the labels in the output file based on the new input file."""
    
    # Read the correct labels from the input file
    correct_labels = {}
    with open(input_file_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            correct_labels[data['index']] = data['label']
    
    # Read the existing output file and update the labels
    updated_data = []
    try:
        with open(output_file_path, 'r') as outfile:
            for line in outfile:
                data = json.loads(line)
                if data['index'] in correct_labels:
                    data['label'] = correct_labels[data['index']]
                updated_data.append(data)
    except FileNotFoundError:
        print("Output file not found. Please ensure the correct file path is provided.")
        return
    
    # Write the updated data back to the output file
    with open(output_file_path, 'w') as outfile:
        for data in updated_data:
            json.dump(data, outfile)
            outfile.write('\n')


if __name__ == "__main__":

    my_api_key = 'b516e173337af7fada2190e8d4b516bb3108'
    os.environ['NCBI_API_KEY'] = my_api_key

    file_path_dataset_stance_binary = '/data/processed/f_primary_binary_1.jsonl'
    file_path_dataset_stance_no_info = '/data/processed/f_primary_no_info_1.jsonl'
    file_path_dataset_stance = '/data/processed/g_primary_shuffle.jsonl'
    file_path_dataset_stance_with_abstracts = '/data/processed/g_primary_publication.jsonl'

    # Shuffle and index files
    shuffle_and_index_files(file_path_dataset_stance_binary, file_path_dataset_stance_no_info, file_path_dataset_stance)

    # Add PubMed abstracts
    add_pubmed_abstracts(file_path_dataset_stance, file_path_dataset_stance_with_abstracts)

    update_labels('/data/processed/g_primary_shuffle.jsonl', '/data/processed/g_primary_publication.jsonl')