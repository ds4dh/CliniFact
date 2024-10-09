import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Function to calculate statistics
def calculate_statistics(df, label_column):
    total_samples = len(df)
    label_counts = df[label_column].value_counts()
    label_distribution = label_counts / total_samples * 100  # Percentage distribution
    return total_samples, label_counts, label_distribution

input_file_path = '../data/processed/g1_primary_publication_ready4split-1.jsonl'

# Load JSONLines Data into DataFrame
data = []
with open(input_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Convert string labels to numeric values
label_map = {'evidence': 1, 'inconclusive': 0, 'no_info': 2}
df['label'] = df['label'].map(label_map)

# Split the Data
random_seed = 1234
train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_seed)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=random_seed)

# Extract Labels and Features
train_labels = train_df['label'].to_numpy()
val_labels = val_df['label'].to_numpy()
test_labels = test_df['label'].to_numpy()

# Features (dataframes are unchanged, so they still contain 'label_stance')
train_features = train_df.copy()
val_features = val_df.copy()
test_features = test_df.copy()

print(f'Average class probability in training set:   {train_labels.mean():.4f}')
print(f'Average class probability in validation set: {val_labels.mean():.4f}')
print(f'Average class probability in test set:       {test_labels.mean():.4f}')

train_total, train_counts, train_distribution = calculate_statistics(train_df, 'label')
val_total, val_counts, val_distribution = calculate_statistics(val_df, 'label')
test_total, test_counts, test_distribution = calculate_statistics(test_df, 'label')

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Set': ['Train', 'Validation', 'Test'],
    'Total Samples': [train_total, val_total, test_total],
    'Label 0 Count': [train_counts.get(0, 0), val_counts.get(0, 0), test_counts.get(0, 0)],
    'Label 1 Count': [train_counts.get(1, 0), val_counts.get(1, 0), test_counts.get(1, 0)],
    'Label 2 Count': [train_counts.get(2, 0), val_counts.get(2, 0), test_counts.get(2, 0)],
    'Label 0 %': [train_distribution.get(0, 0), val_distribution.get(0, 0), test_distribution.get(0, 0)],
    'Label 1 %': [train_distribution.get(1, 0), val_distribution.get(1, 0), test_distribution.get(1, 0)],
    'Label 2 %': [train_distribution.get(2, 0), val_distribution.get(2, 0), test_distribution.get(2, 0)]
})

output_file_path = 'stats_number_of_samples.xlsx'

# Save the DataFrame to an Excel file
summary_df.to_excel(output_file_path, index=False)

# Save Dataframes to CSV (Including 'label_stance')
train_df.to_csv('/data/processed/primary_outcome_publication_dataset/train_set.csv', index=False)
val_df.to_csv('/data/processed/primary_outcome_publication_dataset/validation_set.csv', index=False)
test_df.to_csv('/data/processed/primary_outcome_publication_dataset/test_set.csv', index=False)                                                                                                                                                                                                                                                                                                           