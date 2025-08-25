import zipfile
import glob
import os
import re
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import spacy

# ===========================================================================
### Preparing the COVAREP-features ###
# ===========================================================================

# settings
zip_folder = '../00_dataset_daicwoz'  
dest_folder = '../features_covarep'  

# Regex: File name begins with a number followed by ‘_conv.csv’ (case insensitive)
pattern = re.compile(r'^\d+_COVAREP\.csv$', re.IGNORECASE)


# Find all ZIP files in the folder
zip_files = glob.glob(os.path.join(zip_folder, '*.zip'))
                
# Search every ZIP-file
for zip_pfad in zip_files:
    zip_name = os.path.splitext(os.path.basename(zip_pfad))[0]
    
    try:
        with zipfile.ZipFile(zip_pfad, 'r') as zip_ref:
            # Integrity test of the ZIP file
            if zip_ref.testzip() is not None:
                print(f'ZIP-file defekt: {zip_pfad}')
                continue

            found = False
            for name in zip_ref.namelist():
                basename = os.path.basename(name)
                if pattern.match(basename):
                    dest_filename = f'{basename}' 
                    dest_path = os.path.join(dest_folder, dest_filename)
                    with open(dest_path, 'wb') as f:
                        f.write(zip_ref.read(name))
                    print(f'Extracted: {name} from {zip_pfad} -> {dest_path}')
                    found = True
                    break
            if not found:
                print(f'Not matching file found in {zip_pfad}.')
    except zipfile.BadZipFile:
        print(f'Unreadable ZIP file skipped: {zip_pfad}')

# Naming the columns
# Feature names
covarep_feature_names = [
    "F0", "VUV", "NAQ", "QOQ", "H1H2", "PSP", "MDQ", "peakSlope", "Rd",
    "Rd_conf", "creak",
    *[f"MCEP_{i}" for i in range(25)],
    *[f"HMPDM_{i}" for i in range(25)],
    *[f"HMPDD_{i}" for i in range(13)]
]

assert len(covarep_feature_names) == 74, "Incorrect number of feature names!"

# Eingabeordner
input_folder = "../features_covarep"

# Alle CSV-Dateien durchgehen
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") and not filename.endswith("_aggr.csv"):
        input_path = os.path.join(input_folder, filename)

        # CSV einlesen
        df = pd.read_csv(input_path, header=None)
        df.columns = covarep_feature_names

        # Statistiken berechnen
        quantiles = df.quantile([0.25, 0.5, 0.75])
        skew_vals = df.apply(skew)
        kurt_vals = df.apply(kurtosis)

        aggregated_features = {}
        for col in df.columns:
            aggregated_features[f"{col}_mean"] = df[col].mean()
            aggregated_features[f"{col}_std"] = df[col].std()
            aggregated_features[f"{col}_min"] = df[col].min()
            aggregated_features[f"{col}_max"] = df[col].max()
            aggregated_features[f"{col}_q25"] = quantiles.loc[0.25, col]
            aggregated_features[f"{col}_q50"] = quantiles.loc[0.5, col]
            aggregated_features[f"{col}_q75"] = quantiles.loc[0.75, col]
            aggregated_features[f"{col}_skew"] = skew_vals[col]
            aggregated_features[f"{col}_kurt"] = kurt_vals[col]

        result_df = pd.DataFrame([aggregated_features])

        # Neue Datei speichern
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(input_folder, base + "_aggr.csv")
        result_df.to_csv(output_path, index=False)

        print(f"Processed: {filename} -> {base}_aggr.csv")

print("All files processed!")



# ===========================================================================
### Preparing the FORMANT-features ###
# ===========================================================================

# settings
zip_folder2 = '../00_dataset_daicwoz'  
dest_folder2 = '../features_formant'  

# Regex: File name begins with a number followed by ‘_conv.csv’ (case insensitive)
pattern2 = re.compile(r'^\d+_FORMANT\.csv$', re.IGNORECASE)


# Find all ZIP files in the folder
zip_files2 = glob.glob(os.path.join(zip_folder, '*.zip'))
                
# Search every ZIP-file
for zip_pfad in zip_files:
    zip_name = os.path.splitext(os.path.basename(zip_pfad))[0]
    
    try:
        with zipfile.ZipFile(zip_pfad, 'r') as zip_ref:
            # Integrity test of the ZIP file
            if zip_ref.testzip() is not None:
                print(f'ZIP-file defekt: {zip_pfad}')
                continue

            found = False
            for name in zip_ref.namelist():
                basename = os.path.basename(name)
                if pattern2.match(basename):
                    dest_filename = f'{basename}'  # oder einfach: basename
                    dest_path = os.path.join(dest_folder2, dest_filename)
                    with open(dest_path, 'wb') as f:
                        f.write(zip_ref.read(name))
                    print(f'Extracted: {name} from {zip_pfad} -> {dest_path}')
                    found = True
                    break
            if not found:
                print(f'Not matching file found in {zip_pfad}.')
    except zipfile.BadZipFile:
        print(f'Unreadable ZIP file skipped: {zip_pfad}')
        
# Naming the features
# Feature names
formant_feature_names = [
    "Formant_1", "Formant_2", "Formant_3", "Formant_4", "Formant_5" 
]

assert len(formant_feature_names) == 5, "Incorrect number of features!"

input_folder = "../features_formant"

# Walk through all ZIP-files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") and not filename.endswith("_aggr.csv"):
        input_path = os.path.join(input_folder, filename)

        df = pd.read_csv(input_path, header=None)
        df.columns = formant_feature_names

        quantiles = df.quantile([0.25, 0.5, 0.75])
        skew_vals = df.apply(skew)
        kurt_vals = df.apply(kurtosis)

        aggregated_features_formant = {}
        for col in df.columns:
            aggregated_features_formant[f"{col}_mean"] = df[col].mean()
            aggregated_features_formant[f"{col}_std"] = df[col].std()
            aggregated_features_formant[f"{col}_min"] = df[col].min()
            aggregated_features_formant[f"{col}_max"] = df[col].max()
            aggregated_features_formant[f"{col}_q25"] = quantiles.loc[0.25, col]
            aggregated_features_formant[f"{col}_q50"] = quantiles.loc[0.5, col]
            aggregated_features_formant[f"{col}_q75"] = quantiles.loc[0.75, col]
            aggregated_features_formant[f"{col}_skew"] = skew_vals[col]
            aggregated_features_formant[f"{col}_kurt"] = kurt_vals[col]

        result_df = pd.DataFrame([aggregated_features_formant])

        # Save new file
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(input_folder, base + "_aggr.csv")
        result_df.to_csv(output_path, index=False)

        print(f"Processed: {filename} -> {base}_aggr.csv")

print("All files ready!")


# ===========================================================================
### Text- features ###
# ===========================================================================

# Load English spaCy model
nlp = spacy.load('de_core_news_sm')


# Sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)


# List of filler words (Word Finding Difficulty)
filler_words = ["uh", "um", "er", "ah"]

# Load all interviews and extract patient responses
interviews = []
directory = os.fsencode('../transcripts')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filepath = os.path.join("../transcripts", filename)

    try:
        df = pd.read_csv(filepath, sep='\t', lineterminator='\r')
    except:
        continue  # skip corrupted files

    patient_utterances = df[df['speaker'].str.lower() == 'participant']['value'].dropna().astype(str).tolist()
    if not patient_utterances:
        continue

    full_text = " ".join(patient_utterances)
    participant_id = int(filename.split('_')[0])
    interviews.append({"participant_id": participant_id, "text": full_text})

# Convert to DataFrame
df_patienten = pd.DataFrame(interviews)

# Extract linguistic features
all_features = []

for i, line in df_patienten.iterrows():
    text = line["text"]
    participant_id = line["participant_id"]
    
    doc = nlp(text)

    tokens = [token.text.lower() for token in doc if token.is_alpha]
    types = set(tokens)
    ttr = len(types) / len(tokens) if tokens else 0

    sentence_lengths = [len([token for token in sent]) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    past_verbs = [t for t in doc if t.tag_ in ["VBD", "VBN"]]
    all_verbs = [t for t in doc if t.pos_ == "VERB"]
    past_ratio = len(past_verbs) / len(all_verbs) if all_verbs else 0

    pronouns = [t for t in doc if t.pos_ == "PRON"]

    sent_vectors = [nlp(s.text).vector for s in doc.sents if nlp(s.text).vector_norm > 0]
    if len(sent_vectors) >= 2:
        similarities = [cosine_similarity([sent_vectors[i]], [sent_vectors[i+1]])[0][0] for i in range(len(sent_vectors)-1)]
        mean_coherence = float(np.mean(similarities))
    else:
        mean_coherence = 0

    filler_count = sum(text.lower().count(w) for w in filler_words)

    try:
        sentiment = sentiment_pipeline(text[:512])[0]["label"]
    except:
        sentiment = "N/A"

    features = {
        "participant_id": participant_id,
        "ttr": ttr,
        "avg_sentence_length": avg_sentence_length,
        "past_tense_ratio": past_ratio,
        "pronoun_count": len(pronouns),
        "mean_local_coherence": mean_coherence,
        "filler_word_count": filler_count,
        "sentiment": sentiment
    }

    all_features.append(features)
    print(f"Line {i} is ready.")

# Save results
features_df = pd.DataFrame(all_features)


# ===========================================================================
### Joining all features in one file per participant ###
# ===========================================================================

# Configuration
features_covarep = "features_covarep"
features_formant = "features_formant"
text_features = "text_features.csv"
output_folder = "features_join"

# === Load text features ===
text_df = pd.read_csv(text_features, index_col=0)

# === Collect all participant IDs from audio folders ===
def get_participant_id_from_filename(filename):
    return os.path.basename(filename).split('_')[0]  # Assumes format: 001_xyz.csv

# Mapping participant ID to file paths
features_covarep_files = glob.glob(os.path.join(features_covarep, "*_aggr.csv"))
features_formant_files = glob.glob(os.path.join(features_formant, "*_aggr.csv"))

features_covarep_dict = {get_participant_id_from_filename(f): f for f in features_covarep_files}
features_formant_dict = {get_participant_id_from_filename(f): f for f in features_formant_files}

# === Combine all participant IDs ===
all_participant_ids = set(features_covarep_dict.keys()).intersection(features_formant_dict.keys()).intersection(text_df.index.astype(str))

for pid in sorted(all_participant_ids):
    # Read audio features
    features_covarep_df = pd.read_csv(features_covarep_dict[pid])
    features_formant_df = pd.read_csv(features_formant_dict[pid])

    # Flatten all values (assumes one row per file)
    audio1_features = features_covarep_df.iloc[0].to_dict()
    audio2_features = features_formant_df.iloc[0].to_dict()
    text_features = text_df.loc[int(pid)].to_dict()

    # Combine all features
    combined = {**audio1_features, **audio2_features, **text_features}

    # Create DataFrame with one row of data and one row of headers
    output_df = pd.DataFrame([combined.keys(), combined.values()])
    output_path = os.path.join(output_folder, f"{pid}_joined_features.csv")
    output_df.to_csv(output_path, index=False, header=False)

    print(f"Saved: {output_path}")

print("All participants processed.")



# ===========================================================================
### Conversion to numeric fields only ###
# ===========================================================================

input_dir = "../extracted_features/features_join" 
output_dir = "../extracted_features/features_join_right"

# Function for converting “1 star”, “2 stars”, ... into a number
def convert_star_rating(value):
    match = re.match(r"(\d+)", str(value))
    return int(match.group(1)) if match else None

# Go through all matching files
for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    if os.path.isfile(filepath) and filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        df = pd.read_csv(input_path)

        last_col = df.columns[-1]
        df[last_col] = df[last_col].apply(convert_star_rating)

        df.to_csv(output_path, index=False)
        print(f"Saved converted: {output_path}")



# ===========================================================================
### Combining everything ###
# ===========================================================================

folder1 = "../extracted_features/features_covarep"
folder2 = "../extracted_features/features_formant"
output_folder = "../extracted_features/features_audio_concatenated"

# Helper function: Filter valid CSV files without '_aggr'
def get_valid_files(folder):
    return [f for f in os.listdir(folder)
            if f.endswith('.csv') and not f.endswith('_aggr.csv')]

# Files per folder
files1 = get_valid_files(folder1)
files2 = get_valid_files(folder2)

# Mapping participant-id → file
def build_participant_map(files):
    return {f[:3]: f for f in files if f[:3].isdigit()}

map1 = build_participant_map(files1)
map2 = build_participant_map(files2)

# All shared participant IDs
common_ids = set(map1.keys()) & set(map2.keys())

# Merge by participant ID
for pid in common_ids:
    file1 = os.path.join(folder1, map1[pid])
    file2 = os.path.join(folder2, map2[pid])
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Merge line by line (e.g., using pd.concat with axis=1)
    merged = pd.concat([df1, df2], axis=1)

    # Output
    output_path = os.path.join(output_folder, f"{pid}_combined.csv")
    merged.to_csv(output_path, index=False)

print("Fertig!")
