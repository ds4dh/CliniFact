from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

import os
import argparse
import ast

class StanceDataset(Dataset): 
    def __init__(self, data, tokenizer, use_claim_text=True, truncate_beginning=True): 
        self.data = data
        self.tokenizer = tokenizer
        self.use_claim_text = use_claim_text
        self.truncate_beginning = truncate_beginning
        self.type_mapping = {
            'Superiority': 'superior',
            'Non-Inferiority': 'non-inferior',
            'Equivalence': 'equivalent',
            'Non-Inferiority or Equivalence': 'non-inferior or equivalent',
            'Superiority or Other': 'superior or other'
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        non_inferiority_type_original = ast.literal_eval(item['nonInferiorityTypes'])[0]
        non_inferiority_type = self.type_mapping[non_inferiority_type_original]

        if self.use_claim_text:
            text = f"{item['intervention_group_title']} is {non_inferiority_type} to {item['comparator_group_title']} in terms of {item['outcome_title']} {item['outcome_timeFrame']}"
        else:
            text = f"{item['outcome_title']} {item['outcome_timeFrame']} {item['intervention_group_title']} {non_inferiority_type_original} {item['comparator_group_title']}"
        text_pair = f"{item['article_title']} {item['article_abstract']}"

        if self.truncate_beginning:
            encoded_length = len(self.tokenizer.encode(text, text_pair, add_special_tokens=True))
            while encoded_length > 512:
                text_pair_sentences = text_pair.split('. ')
                if len(text_pair_sentences) > 1:
                    # Remove the first sentence
                    text_pair = '. '.join(text_pair_sentences[1:])
                else:
                    # Word-level truncation for a single long sentence
                    text_pair_words = text_pair.split()
                    if len(text_pair_words) > 15: # not sure if we should use the cutoff of 1 word, maybe 15 words are better, and the description of the primary outcome is too long, we use automatic truncation to do that
                        text_pair = ' '.join(text_pair_words[1:])
                    else:
                        break  # Break if there's only one word left
                encoded_length = len(self.tokenizer.encode(text, text_pair, add_special_tokens=True))

            encoded_item = self.tokenizer.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
        else:
            encoded_item = self.tokenizer.encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
        
        return {
            'input_ids': encoded_item['input_ids'].flatten(),
            'attention_mask': encoded_item['attention_mask'].flatten()
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, required=True, 
                        help="Specify the model folder (e.g., 'models-claim-text' or 'models-plain-text'")
    parser.add_argument('--use_claim_text', type=bool, default=True,
                    help="Specify whether to use claim text (default: True)")
    parser.add_argument('--file_type', type=str, default='plain_text', 
                        help="Specify the file type part of the file name (e.g., 'plain_text', 'claim_text')")
    parser.add_argument('--truncate_beginning', type=bool, default=True,
                    help="Specify whether to truncate at the beginning of the abstract (default: True)")
    parser.add_argument('--cuda_device_id', type=int, default=0,
                    help="Specify the CUDA device ID to use (default: 0)")
    args = parser.parse_args()


    test_file_path = './data/primary_outcome_publication_dataset/test_set.csv'
    df_test = pd.read_csv(test_file_path)

    pretrained_model_names_or_paths = [
        'google-bert/bert-base-uncased',
        'FacebookAI/roberta-base', 
        'sentence-transformers/multi-qa-mpnet-base-dot-v1', 
        'dmis-lab/biobert-base-cased-v1.1', 
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    ]

    for model_name in pretrained_model_names_or_paths:

        print(f'Using model: {model_name}')

        model_folder = args.model_folder

        # Define the paths to save the model and tokenizer
        model_read_path = f"./{model_folder}/save-model-{model_name}"
        tokenizer_read_path = f"./{model_folder}/save-tokenizer-{model_name}"

        # Define the device
        cuda_device_id = args.cuda_device_id
        device = torch.device(f'cuda:{cuda_device_id}') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {device}')

        # Create the dataloaders
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_read_path)

        test_dataset = StanceDataset(df_test, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(model_read_path, num_labels=3)
        model = model.to(device)

        # Evaluate the model on the validation set
        model.eval()

        test_preds = []
        
        for batch in test_dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                test_preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())

        # Add the 'Predicted_Label' column to the existing DataFrame
        df_test['Predicted_Label'] = test_preds
        file_path = f"prediction_{model_name}_{args.file_type}.csv"
        directory = os.path.dirname(file_path)

        if not os.path.exists(directory):
            os.makedirs(directory)
            
        df_test.to_csv(file_path, index=False)
