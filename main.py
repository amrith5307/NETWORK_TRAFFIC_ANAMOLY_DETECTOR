import pandas as pd
import numpy as np
import os
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.isolation_forest_model import IsolationForestModel 
from src.unsupervised_pro import HBOSDetector, EnvelopeDetector 
from src.fusion import AnomalyFuser
from src.visualization import plot_final_comparison, plot_confusion_matrix

def main():
    # 1. ROBUST FOLDER SETUP
    base_dir = "results"
    plots_dir = os.path.join(base_dir, "plots")
    
    # Clean up old results carefully
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    
    os.makedirs(plots_dir, exist_ok=True)

    # 2. LOAD DATA
    train_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_train.csv"
    test_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_test.csv"
    
    print("Loading data...")
    train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    # 3. PREPROCESS
    X_train, y_train, encoders = preprocess_data(train_raw, is_train=True)
    X_test, y_test = preprocess_data(test_raw, is_train=False, encoder_dict=encoders)

    # 4. STABLE SAMPLING (30k rows)
    sample_size = 30000
    idx = np.random.choice(len(X_train), min(len(X_train), sample_size), replace=False)
    X_train_sample = X_train[idx]

    # 5. RUN MODELS (Weighted Logic)
    iso_model = IsolationForestModel(n_estimators=100, contamination=0.40)
    iso_model.train(X_train_sample)
    iso_pred = iso_model.evaluate(X_test, y_test)

    hbos_model = HBOSDetector(contamination=0.40)
    hbos_model.train(X_train_sample)
    hbos_pred = hbos_model.predict(X_test)

    env_model = EnvelopeDetector(contamination=0.40)
    env_model.train(X_train_sample)
    env_pred = env_model.predict(X_test)

    # 6. WEIGHTED FUSION
    fuser = AnomalyFuser([iso_pred, hbos_pred, env_pred])
    final_pred = fuser.weighted_vote() 

    # 7. EXPORT RESULTS
    models = [("IsoForest", iso_pred), ("HBOS", hbos_pred), ("Envelope", env_pred), ("ENSEMBLE", final_pred)]
    summary = []
    for name, pred in models:
        summary.append({
            "MODEL": name,
            "ACCURACY": accuracy_score(y_test, pred),
            "PRECISION": precision_score(y_test, pred, zero_division=0),
            "RECALL": recall_score(y_test, pred, zero_division=0),
            "F1-SCORE": f1_score(y_test, pred, zero_division=0)
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_dir, "latest_results.csv"), index=False)
    
    plot_final_comparison(summary)
    plot_confusion_matrix(y_test, final_pred, "ENSEMBLE")
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()