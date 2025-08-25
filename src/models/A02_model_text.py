import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F

# PARAMETERS
VAL_CSV = '00_dataset_daicwoz/dev_split_Depression_AVEC2017.csv'
TEST_CSV = '00_dataset_daicwoz/full_test_split.csv'
TRANSCRIPTS_DIR = 'transcripts'
BATCH_SIZE = 16
DROPOUT = 0.4
CHUNK_LENGTH = 512
CHUNK_STRIDE = 256

# Prepare validation dataframe
val_file = pd.read_csv(VAL_CSV)
test_file = pd.read_csv(TEST_CSV)
label_dict = {0: 'cheerful', 1: 'depressed'}

# Build interview dataframe
interviews = []
directory = os.fsencode(TRANSCRIPTS_DIR)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    interview = []
    transcript_file = pd.read_csv(os.path.join(TRANSCRIPTS_DIR, filename), sep='\t', lineterminator='\r')
    for _, row in transcript_file.iterrows():
        speaker = row.get('speaker', '')
        value = row.get('value', '')
        if speaker and value:
            interview.append(f"{speaker}: ({value})")
    interviews.append([int(filename.split('_')[0]), ". ".join(interview)])

df = pd.DataFrame(interviews, columns=['Participant_ID', 'interview'])

# Merge with validation & test labels
val_subset = val_file[['Participant_ID', 'PHQ8_Binary']]
val_df = pd.merge(val_subset, df, on='Participant_ID', how='inner')

test_subset = test_file[['Participant_ID', 'PHQ_Binary']]
test_df = pd.merge(test_subset, df, on='Participant_ID', how='inner')
test_df = test_df.rename(columns={'PHQ_Binary': 'PHQ8_Binary'})

# Dataset class for validation
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.samples = []

        for _, row in df.iterrows():
            text = row['interview']
            label = row['PHQ8_Binary']
            participant_id = row['Participant_ID']

            # Encode full text (no truncation)
            tokens = tokenizer.encode(text, add_special_tokens=True)

            # Chunk the token list
            for i in range(0, len(tokens), max_length - stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) < 10:
                    continue
                self.samples.append({
                    'input_ids': chunk,
                    'label': label,
                    'participant_id': participant_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = sample['input_ids']
        label = sample['label']
        participant_id = sample['participant_id']

        encoding = self.tokenizer.prepare_for_model(
            input_ids,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['participant_id'] = torch.tensor(participant_id, dtype=torch.long)
        return item

# Tokenizer, model, and DataLoader for validation
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

val_dataset = TextDataset(val_df, tokenizer, CHUNK_LENGTH, CHUNK_STRIDE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_dataset = TextDataset(test_df, tokenizer, CHUNK_LENGTH, CHUNK_STRIDE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def get_text_model():
    config = BertConfig.from_pretrained(model_name, hidden_dropout_prob=DROPOUT, attention_probs_dropout_prob=DROPOUT)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    checkpoint = torch.load("best_text_model.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_text_val_loader():
    return val_loader

def get_text_test_loader():
    return test_loader

def get_text_preds(loader, device='cpu'):
    model = get_text_model().to(device)
    preds, labels, participant_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ['labels', 'participant_id']}
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds.extend(probs)
            labels.extend(batch['labels'].cpu().numpy())
            participant_ids.extend(batch['participant_id'].cpu().numpy())
    return np.array(preds), np.array(labels), np.array(participant_ids)