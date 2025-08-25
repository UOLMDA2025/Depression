import os
import numpy as np
import librosa
import librosa.effects
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from scipy.signal import resample
import random
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch, Gain, HighPassFilter
import noisereduce as nr

# PARAMETERS
AUDIO_DIR = '01_audio'
CHUNK_DURATION = 30
CHUNK_STRIDE = 0.5
SAMPLE_RATE = 16000
BATCH_SIZE = 4

# Load CSVs and build label dicts
import pandas as pd
val_file = pd.read_csv('00_dataset_daicwoz/dev_split_Depression_AVEC2017.csv')
val_label_dict = val_file.set_index('Participant_ID')['PHQ8_Binary'].to_dict()

test_file = pd.read_csv('00_dataset_daicwoz/full_test_split.csv')
test_label_dict = test_file.set_index('Participant_ID')['PHQ_Binary'].to_dict()

# Match audio files to val split
import glob
all_files = glob.glob(os.path.join(AUDIO_DIR, '*'))
val_files = []
test_files = []
for file in all_files:
    basename = os.path.basename(file)
    participant_id_str = basename.split('_')[0]
    try:
        participant_id = int(participant_id_str)
    except ValueError:
        continue
    if participant_id in val_label_dict:
        val_files.append(file)
    elif participant_id in test_label_dict:
        test_files.append(file)

# Dataset class (no augmentation needed for val)
class AudioChunkDataset(Dataset):
    def __init__(self, file_list, label_dict, chunk_duration, sample_rate, mode='train', n_chunks=1):
        self.file_list = file_list
        self.label_dict = label_dict
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.mode = mode
        self.n_chunks = n_chunks
        self.index_map = []

        chunk_size = int(chunk_duration * sample_rate)
        stride = int(chunk_size * CHUNK_STRIDE)
        
        if self.mode == 'train':
            self.augmenter = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.2),
                Gain(min_gain_db=-5.0, max_gain_db=10.0, p=0.3),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
                TimeStretch(min_rate=0.90, max_rate=1.1, p=0.3),
                HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=300, p=0.4),
            ])

        for file_path in self.file_list:
            y, sr = librosa.load(file_path, sr=sample_rate)
            total_samples = len(y)

            if self.mode == 'train':
                for _ in range(n_chunks):
                    self.index_map.append((file_path, None))
            else:
                starts = list(range(0, total_samples - chunk_size + 1, stride))
                if not starts:
                    starts = [0]
                for i, _ in enumerate(starts):
                    self.index_map.append((file_path, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, chunk_idx = self.index_map[idx]
        participant_id = int(os.path.basename(file_path).split('_')[0])
        label = self.label_dict[participant_id]

        y, sr = librosa.load(file_path, sr=self.sample_rate)
        chunk_samples = int(self.chunk_duration * sr)

        if len(y) < chunk_samples:
            y = np.pad(y, (0, chunk_samples - len(y)))
            start = 0
        else:
            if self.mode == 'train':
                max_start = len(y) - chunk_samples
                start = np.random.randint(0, max_start + 1)
            else:
                stride = int(chunk_samples * CHUNK_STRIDE)
                start = chunk_idx * stride
                start = min(start, len(y) - chunk_samples)

        y = y[start:start + chunk_samples]
        if self.mode != 'train':
            y = nr.reduce_noise(y=y, sr=self.sample_rate)
        y = self.normalize_volume(y)

        if self.mode == 'train':
            y = self.augmenter(samples=y, sample_rate=self.sample_rate)

        return torch.tensor(y, dtype=torch.float32), label, participant_id
    
    def normalize_volume(self, y, target_dBFS=-20):
        rms = np.sqrt(np.mean(y**2))
        scalar = 10 ** (target_dBFS / 20) / (rms + 1e-6)
        y = y * scalar
        return np.clip(y, -1.0, 1.0)

# Feature extractor and model class
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

class Wav2Vec2Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=2,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True
        )

    def forward(self, input_values, attention_mask=None):
        return self.model(input_values=input_values, attention_mask=attention_mask).logits

# Collate function
def collate_fn(batch):
    audios, labels, participant_ids = zip(*batch)
    audios = [a.numpy() for a in audios]
    inputs = feature_extractor(
        audios,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    labels = torch.tensor(labels, dtype=torch.long)
    participant_ids = torch.tensor(participant_ids, dtype=torch.long)
    return {**inputs, "labels": labels, "participant_ids": participant_ids}

# Validation DataLoader
val_dataset = AudioChunkDataset(val_files, val_label_dict, CHUNK_DURATION, SAMPLE_RATE, mode='val')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

test_dataset = AudioChunkDataset(test_files, test_label_dict, CHUNK_DURATION, SAMPLE_RATE, mode='test')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Inference functions for late fusion
def get_audio_model():
    model = Wav2Vec2Classifier()
    checkpoint = torch.load("01_best_audio_model.pth", map_location='cuda', weights_only=False)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_audio_val_loader():
    return val_loader

def get_audio_test_loader():
    return test_loader

def get_audio_preds(loader, device='cpu'):
    model = get_audio_model().to(device)
    preds, labels, part_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_values', 'attention_mask']}
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=-1)[:, 1].cpu().numpy()
            preds.extend(probs)
            labels.extend(batch['labels'].cpu().numpy())
            part_ids.extend(batch['participant_ids'].cpu().numpy())
    return np.array(preds), np.array(labels), np.array(part_ids)