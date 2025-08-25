# DepressNet: Multimodal Depression Detection Using DAIC-WOZ

This repository contains the code for **DepressNet**, a multimodal deep learning framework for automatic depression detection based on audio, text, and tabular features extracted from clinical interviews. The system is evaluated using the publicly available **DAIC-WOZ** dataset.

> This project was conducted as part of the AI4Health initiative at Carl von Ossietzky University Oldenburg.

---

## Project Overview

**Objective**:  
To support early detection of depressive symptoms by integrating multiple modalities using machine learning.

**Modalities Used**:

- **Audio**: Raw speech signals processed with Wav2Vec2.
- **Text**: Transcripts of patient responses, analyzed using BERT.
- **Tabular features**: Acoustic (Covarep, formants) and linguistic features (e.g., TTR, past tense ratio, sentiment) extracted from the audio and text.

**Fusion Method**:

- Late fusion using a logistic regression classifier trained on calibrated modality-specific outputs.

---

## Repository Structure

```bash
.
├── fusion/                    # Late fusion and evaluation logic
│   └── train_evaluate.py
│
├── models/                   # Scripts to load and predict with each modality
│   ├── A01_model_audio.py
│   ├── A02_model_text.py
│   └── A03_model_tab_features.py
│
├── notebooks/                # Jupyter notebooks for development and exploration
│   ├── model01_audio.ipynb
│   ├── model02_text.ipynb
│   ├── model03_tab_features.ipynb
│   └── model_late_fusion.ipynb
│
├── saved_models/             # Trained models for each modality
│   ├── 01_best_audio_model.pth
│   ├── 02_best_text_model.pt
│   └── 03_best_tabular_model.pkl
│
── data_preparation/             # Scripts for data preparation
│   ├── tab_feature_preparation.py
│
├── results/                  # Output plots and evaluation metrics
│   └── (created automatically)
│
├── requirements.txt
├── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

Due to licensing restrictions, the DAIC-WOZ dataset is **not included** in this repository.  
To run the model, request access via: https://dcapswoz.ict.usc.edu/

The original DAIC-WOZ dataset contains one `.zip` file per participant (e.g., `302_P.zip`).  
Each ZIP archive includes:

- `XXX_AUDIO.wav` – the audio recording used for the **audio model**
- `XXX_TRANSCRIPT.csv` – the dialogue transcript used for the **text model**

After downloading, extract the contents of each `.zip` file into:

```bash
data/
├── audio/         # Contains all XXX_AUDIO.wav files
├── transcripts/   # Contains all XXX_TRANSCRIPT.csv files
```

Additionally, the dataset provides predefined train/val/test splits:

- `train_split_Depression_AVEC2017.csv` – used as training set
- `dev_split_Depression_AVEC2017.csv` – used as validation set
- `full_test_split.csv` – used as test set (with labels)

These official splits are used exactly as provided for all experiments in this repository.

### 3. Preprocess the tabular data

For preparing the tabular features, run the script data_preparation/tab_feature_preparation.py. The tabular features will be extracted and calculated in the right form and saved.

### 4. Run late fusion evaluation

```bash
python fusion/train_evaluate.py
```

This will:

- Load modality-specific predictions on validation and test sets
- Calibrate each modality (isotonic regression)
- Train logistic regression fusion on validation set
- Evaluate on test set
- Output performance metrics and plots to `results/`

---

## Results

The final multimodal model achieved the following performance on the DAIC-WOZ test set:

| Metric         | Value |
|----------------|-------|
| AUROC          | 0.88  |
| Macro F1-score | 0.75  |
| Positive F1    | 0.64  |
| Negative F1    | 0.87  |

The model is well-calibrated (calibration error ≈ 0.04) and shows clinical net benefit over "refer all" and "refer none" strategies.

---

## Use Cases

- **Clinical screening support** (as part of a multi-stage diagnostic pipeline)
- **AI research** in multimodal learning and neuropsychiatric ML
- **Educational** example of fusion architectures using real-world data

---

##  Ethical & Legal Notes

- **Do not upload DAIC-WOZ data** to this repository.
- This model is **not intended for clinical deployment** without further validation.
- Follow ethical guidelines when using ML in mental health contexts.

---

## License

This code is released under the **MIT License**.  
The dataset (DAIC-WOZ) must be obtained and used in compliance with its own license.

---

## Authors

This work was conducted by the AI4Health Division, Carl von Ossietzky University Oldenburg.

- Jana Weber
- Marcel Weber
- Juan Miguel Lopez Alcaraz
- Nils Strodthoff

For more details, please refer to the preprint:  
[DepressNet: Depression diagnosis via deep multimodal feature fusion from patient interviews](https://github.com/AI4HealthUOL/DepressNet)