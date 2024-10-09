from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score,f1_score
import time
import os
import ast
import argparse

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
            'attention_mask': encoded_item['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
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

    pretrained_model_names_or_paths = [
    'google-bert/bert-base-uncased',
    'FacebookAI/roberta-base', 
    'sentence-transformers/multi-qa-mpnet-base-dot-v1', 
    'dmis-lab/biobert-base-cased-v1.1', 
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    ]

    # Load the training and validation datasets
    train_file_path = './data/primary_outcome_publication_dataset/train_set.csv'
    val_file_path = './data/primary_outcome_publication_dataset/validation_set.csv'

    df_train = pd.read_csv(train_file_path)
    df_val = pd.read_csv(val_file_path)

    for model_name in pretrained_model_names_or_paths:

        # Define the model name
        print(f'Using model: {model_name}')
        model_folder = args.model_folder #'models-claim-text' #'models-plain-text'

        # Define the paths to save the model and tokenizer
        model_save_path = f"../{model_folder}/save-model-{model_name}"
        tokenizer_save_path = f"../{model_folder}/save-tokenizer-{model_name}"

        # Define the device
        cuda_device_id = args.cuda_device_id
        device = torch.device(f'cuda:{cuda_device_id}') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {device}')

        # Create the dataloaders
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = StanceDataset(df_train, tokenizer, use_claim_text=args.use_claim_text, truncate_beginning=args.truncate_beginning)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # the batch size depends on the model

        dev_dataset = StanceDataset(df_val, tokenizer, use_claim_text=args.use_claim_text, truncate_beginning=args.truncate_beginning)
        dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=False)

        # Load pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        model = model.to(device)

        # Define the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=2e-5)

        # Define the training loop
        grad_clip = 1.0
        early_stopping_patience = 10
        best_val_f1_macro = 0
        best_epoch = 0
        patience_counter = 0

        # Create a DataFrame to store the training metrics
        metrics_df = pd.DataFrame(columns=['Epoch', 'Average Training Loss', 'Training Accuracy', 'Training F1 Macro',
                                        'Validation Loss', 'Validation Accuracy', 'Validation F1 Macro', 'Epoch Duration'])

        # Training loop
        for epoch in range(100): # 100 epochs
            model.train()
            start_time = time.time() # start time for the epoch
        
            total_loss = 0.0
            all_labels = []
            all_preds = []
            num_batches = len(train_dataloader)

            for batch in train_dataloader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                total_loss += loss.item()  # Accumulate the loss

                preds = torch.argmax(outputs.logits, axis=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Compute the average loss, accuracy, and F1 score
            average_loss = total_loss / num_batches
            training_accuracy = accuracy_score(all_labels, all_preds)
            training_f1_marco = f1_score(all_labels, all_preds, average='macro')

            end_time = time.time()  # End time of the epoch
            epoch_duration = end_time - start_time  # Duration of the epoch

            # Print metrics
            print(f"Epoch {epoch + 1}/{100}, Duration: {epoch_duration:.2f} seconds, "
                f"Average Training Loss: {average_loss:.4f}, Training Accuracy: {training_accuracy:.4f}, Training F1 Macro: {training_f1_marco:.4f}")

            # Evaluate the model on the validation set
            model.eval()

            val_losses = []
            val_preds = []
            val_labels = []
            
            for batch in dev_dataloader:
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                    val_losses.append(outputs.loss.item())
                    val_preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_f1_macro = f1_score(val_labels, val_preds, average='macro')

            if val_f1_macro > best_val_f1_macro:
                best_val_f1_macro = val_f1_macro
                patience_counter = 0

                model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(tokenizer_save_path)
                print(f"Epoch {epoch + 1}: Validation F1-macro improved, saving model.")

            else:
                patience_counter += 1

            # Print metrics
            print(f"Epoch {epoch + 1}/{100}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Macro: {val_f1_macro:.4f}")

            # Append metrics to the DataFrame
            metrics_df.loc[epoch] = {
                'Epoch': epoch + 1,
                'Average Training Loss': round(average_loss, 4),
                'Training Accuracy': round(training_accuracy, 4),
                'Training F1 Macro': round(training_f1_marco, 4), 
                'Epoch Duration': round(epoch_duration, 4),
                'Validation Loss': round(avg_val_loss, 4),
                'Validation Accuracy': round(val_accuracy, 4),
                'Validation F1 Macro': round(val_f1_macro, 4)
            }
            
            if patience_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation F1-score.")
                break        

        file_path = f"training_metrics_{args.file_type}_{model_name}.csv"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        metrics_df.to_csv(file_path, index=False)