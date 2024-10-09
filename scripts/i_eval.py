import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, required=True, 
                        help="Specify the model folder (e.g., 'models-claim-text' or 'models-plain-text'")
    parser.add_argument('--file_type', type=str, default='plain_text', 
                        help="Specify the file type part of the file name (e.g., 'plain_text', 'claim_text')")
    args = parser.parse_args()
    
    pretrained_model_names_or_paths = [
        'google-bert/bert-base-uncased',
        'FacebookAI/roberta-base', 
        'sentence-transformers/multi-qa-mpnet-base-dot-v1', 
        'dmis-lab/biobert-base-cased-v1.1', 
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    ]

    for model_name in pretrained_model_names_or_paths:    
        file_path = f"prediction_{model_name}_{args.file_type}.csv"
        results_df = pd.read_csv(file_path)

        # Extract true and predicted labels
        true_labels = results_df['label']
        predicted_labels = results_df['Predicted_Label']

        # Generate the classification report as a dictionary
        report_dict = classification_report(true_labels, predicted_labels)

        print(model_name)
        print(report_dict)

