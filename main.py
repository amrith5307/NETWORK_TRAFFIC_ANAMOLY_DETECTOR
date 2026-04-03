import pandas as pd
import numpy as np
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Local imports
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.isolation_forest_model import IsolationForestModel
from src.unsupervised_pro import EnvelopeDetector
from src.fusion import AnomalyFuser
from src.visualization import plot_final_comparison, plot_confusion_matrix

def main():
    base_dir = "results"
    plots_dir = os.path.join(base_dir, "plots")

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(plots_dir, exist_ok=True)

    # LOAD DATA
    train_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_train.csv"
    test_path  = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_test.csv"

    print("Loading data...")
    train_raw = load_data(train_path)
    test_raw  = load_data(test_path)

    # PREPROCESS (No PCA anymore)
    X_train, y_train, encoders = preprocess_data(train_raw, is_train=True)
    X_test,  y_test             = preprocess_data(test_raw, is_train=False, encoder_dict=encoders)

    contamination = 0.46

    # TRAIN MODELS
    print("Training Isolation Forest...")
    iso_model = IsolationForestModel(n_estimators=200, contamination=contamination)
    iso_model.train(X_train)

    print("Training Elliptic Envelope...")
    env_model = EnvelopeDetector(contamination=contamination)
    env_model.train(X_train)

    # Ensemble Threshold Tuning
    X_val, X_eval, y_val, y_eval = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    fuser = AnomalyFuser(iso_model, env_model, iso_weight=0.4, env_weight=0.6)
    print("Tuning ensemble threshold on full feature set...")
    fuser.tune_threshold(X_val, y_val)

    # EVALUATION
    final_pred_eval = fuser.predict(X_eval)
    iso_pred_eval = iso_model.predict(X_eval)
    env_pred_eval = env_model.predict(X_eval)

    # RESULTS
    models = [
        ("IsoForest", iso_pred_eval),
        ("Envelope",  env_pred_eval),
        ("ENSEMBLE",  final_pred_eval)
    ]

    summary = []
    for name, pred in models:
        summary.append({
            "MODEL":     name,
            "ACCURACY":  accuracy_score(y_eval, pred),
            "PRECISION": precision_score(y_eval, pred, zero_division=0),
            "RECALL":    recall_score(y_eval, pred, zero_division=0),
            "F1-SCORE":  f1_score(y_eval, pred, zero_division=0)
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_dir, "latest_results.csv"), index=False)

    plot_final_comparison(summary)
    plot_confusion_matrix(y_eval, final_pred_eval, "ENSEMBLE")

    print("\n=== FINAL RESULTS (Full Feature Space) ===")
    print(summary_df.to_string(index=False))
    print("\nPipeline Complete.")

if __name__ == "__main__":
    main()