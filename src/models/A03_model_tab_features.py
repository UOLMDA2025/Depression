import os
import glob
import pandas as pd
import numpy as np
import joblib

# Constants
VAL_CSV = '00_dataset_daicwoz/dev_split_Depression_AVEC2017.csv'
TEST_CSV = '00_dataset_daicwoz/full_test_split.csv'
FEATURE_DIR = 'extracted_features/features_join_right'
MODEL_PATH = 'best_tabular_model.pkl'


# Load label mapping
val_file = pd.read_csv(VAL_CSV)
val_label_dict = val_file.set_index('Participant_ID')['PHQ8_Binary'].to_dict()

test_file = pd.read_csv(TEST_CSV)
test_label_dict = test_file.set_index('Participant_ID')['PHQ_Binary'].to_dict()

# Match feature files to val set
all_files = glob.glob(os.path.join(FEATURE_DIR, '*'))
val_items = []
for file in all_files:
    try:
        participant_id = int(os.path.basename(file).split('_')[0])
    except ValueError:
        continue
    if participant_id in val_label_dict:
        val_items.append((file, val_label_dict[participant_id]))

# Load features and labels
val_df = pd.concat([
    pd.read_csv(file).assign(label=label, participant_id=int(os.path.basename(file).split('_')[0]))
    for file, label in val_items
], ignore_index=True)

X_val = val_df.drop(columns=["label", "participant_id"])
y_val = val_df["label"]
participant_ids = val_df["participant_id"]

# Clean NaNs/Infs
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.mean())

# Load trained model
def get_tabular_model():
    model = joblib.load(MODEL_PATH)
    return model

# Inference
def get_tabular_preds():
    # Load validation labels
    val_file = pd.read_csv(VAL_CSV)
    val_label_dict = val_file.set_index('Participant_ID')['PHQ8_Binary'].to_dict()

    # Collect matching files
    all_files = glob.glob(os.path.join(FEATURE_DIR, '*'))
    val_items = []
    for file in all_files:
        try:
            participant_id = int(os.path.basename(file).split('_')[0])
        except ValueError:
            continue
        if participant_id in val_label_dict:
            val_items.append((file, val_label_dict[participant_id]))

    # Load features into DataFrame
    val_df = pd.concat([
        pd.read_csv(file).assign(label=label, participant_id=int(os.path.basename(file).split('_')[0]))
        for file, label in val_items
    ], ignore_index=True)

    # Extract features and labels
    X_val = val_df.drop(columns=["label", "participant_id"])
    y_val = val_df["label"]
    participant_ids = val_df["participant_id"]

    # Clean NaNs and Infs
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_val.mean())

    # Load pre-trained tabular model
    model = joblib.load(MODEL_PATH)

    # Predict probabilities
    probs = model.predict_proba(X_val)[:, 1]  # class 1 probability

    return probs, y_val.values, participant_ids.values

def get_tabular_test_preds():
    # Collect test files matching participant IDs
    all_files = glob.glob(os.path.join(FEATURE_DIR, '*'))
    test_items = []
    for file in all_files:
        try:
            participant_id = int(os.path.basename(file).split('_')[0])
        except ValueError:
            continue
        if participant_id in test_label_dict:
            test_items.append((file, test_label_dict[participant_id]))

    # Load test features into DataFrame
    test_df = pd.concat([
        pd.read_csv(file).assign(label=label, participant_id=int(os.path.basename(file).split('_')[0]))
        for file, label in test_items
    ], ignore_index=True)

    # Extract features and labels
    X_test = test_df.drop(columns=["label", "participant_id"])
    y_test = test_df["label"]
    participant_ids = test_df["participant_id"]

    # Clean NaNs and Infs
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.mean())

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]  # class 1 probability

    return probs, y_test.values, participant_ids.values