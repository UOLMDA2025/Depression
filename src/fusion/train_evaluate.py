
import os
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix, RocCurveDisplay
from models.A01_model_audio import load_audio_predictions
from models.A02_model_text import load_text_predictions
from models.A03_model_tab_features import load_tabular_predictions
import seaborn as sns

os.makedirs("results", exist_ok=True)


def evaluate_model(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)
    cm = confusion_matrix(y_true, y_pred)
    return auc, macro_f1, f1_pos, f1_neg, cm


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.grid()
    plt.savefig("results/auroc.png")
    plt.close()


def plot_reliability_diagram(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("Calibration Plot")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.grid()
    plt.savefig("results/calibration.png")
    plt.close()


def plot_probability_scatter(y_true, y_prob):
    errors = np.abs(y_true - y_prob)
    plt.figure()
    sns.scatterplot(x=y_prob, y=errors)
    plt.title("Error vs Predicted Probability")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Absolute Error")
    plt.grid()
    plt.savefig("results/net_benefit.png")
    plt.close()


def main():
    # Load predictions and labels from each modality
    audio_val, audio_val_labels = load_audio_predictions(split='val')
    text_val, text_val_labels = load_text_predictions(split='val')
    tab_val, tab_val_labels = load_tabular_predictions(split='val')

    audio_test, audio_test_labels = load_audio_predictions(split='test')
    text_test, text_test_labels = load_text_predictions(split='test')
    tab_test, tab_test_labels = load_tabular_predictions(split='test')

    # Stack predictions for fusion
    X_val = np.vstack([audio_val, text_val, tab_val]).T
    X_test = np.vstack([audio_test, text_test, tab_test]).T

    # Ensure labels match
    assert (audio_val_labels == text_val_labels).all() and (audio_val_labels == tab_val_labels).all()
    assert (audio_test_labels == text_test_labels).all() and (audio_test_labels == tab_test_labels).all()
    y_val = audio_val_labels
    y_test = audio_test_labels

    # Fit calibrated fusion model
    base_model = LogisticRegression()
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    base_model.fit(X_val, y_val)
    calibrated_model.fit(X_val, y_val)

    # Predict on test set
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    auc, macro_f1, f1_pos, f1_neg, cm = evaluate_model(y_test, test_probs)

    # Print results
    print("Test AUROC:", auc)
    print("Macro F1-score:", macro_f1)
    print("Positive F1-score:", f1_pos)
    print("Negative F1-score:", f1_neg)
    print("Confusion Matrix:\n", cm)

    # Save model
    joblib.dump(calibrated_model, "results/late_fusion_model.joblib")

    # Generate plots
    plot_roc(y_test, test_probs)
    plot_reliability_diagram(y_test, test_probs)
    plot_probability_scatter(y_test, test_probs)


if __name__ == "__main__":
    main()
