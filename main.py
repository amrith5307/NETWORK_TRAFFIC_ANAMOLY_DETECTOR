import pandas as pd
import os
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.kmeans_model import KMeansAnomalyDetector
from src.isolation_forest_model import IsolationForestModel
from src.dbscan_model import DBSCANAnomalyDetector
from src.fusion import AnomalyFuser
from src.visualization import plot_final_comparison, plot_clusters

def main():
    # 1. CLEANUP & DIRECTORY SETUP
    # This ensures no old "ghost" files interfere with your new results
    if os.path.exists("results"):
        print("Cleaning up old results directory...")
        shutil.rmtree("results")
    
    os.makedirs("results/plots", exist_ok=True)

    print("\n" + "═"*60)
    print(" PHASE 1: LOADING & SYNCHRONIZED PREPROCESSING ".center(60, "═"))
    print("═"*60)

    train_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_train.csv"
    test_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_test.csv"

    try:
        # Load Raw Data
        train_raw = load_data(train_path)
        test_raw = load_data(test_path)

        # Preprocess Train (is_train=True returns the encoder mapping)
        X_train, y_train, encoders = preprocess_data(train_raw, is_train=True)
        
        # Preprocess Test (is_train=False uses the same encoder mapping to prevent feature mismatch)
        X_test, y_test = preprocess_data(test_raw, is_train=False, encoder_dict=encoders)

        print(f" Preprocessing Success!")
        print(f"Features in Train: {X_train.shape[1]} | Features in Test: {X_test.shape[1]}")
        
        # Generate the Cluster Map (for Tab 4)
        plot_clusters(X_train, y_train, title="Network Traffic Distribution")

    except Exception as e:
        print(f"ERROR in Phase 1: {e}")
        return

    print("\n" + "═"*60)
    print(" PHASE 2: TRAINING AI MODELS ".center(60, "═"))
    print("═"*60)

    # 1. Isolation Forest
    print("Training Isolation Forest...")
    iso_model = IsolationForestModel(n_estimators=100, contamination=0.2)
    iso_model.train(X_train)
    iso_pred = iso_model.evaluate(X_test, y_test) 

    # 2. KMeans
    print("Training KMeans...")
    km_model = KMeansAnomalyDetector(n_clusters=8)
    km_model.train(X_train)
    km_pred = km_model.evaluate(X_test, y_test)

    # 3. DBSCAN (Unsupervised - uses test data for immediate detection)
    print("Running DBSCAN Anomaly Detection...")
    db_model = DBSCANAnomalyDetector(eps=0.5, min_samples=5)
    db_pred = db_model.evaluate(X_test, y_test)

    # 4. Ensemble Fusion (Majority Voting)
    print("Fusing models into Ensemble...")
    fuser = AnomalyFuser([iso_pred, km_pred, db_pred])
    final_pred = fuser.majority_vote()

    # --- FINAL REPORTING & EXPORT ---
    models = [
        ("Isolation Forest", iso_pred),
        ("KMeans", km_pred),
        ("DBSCAN", db_pred),
        ("Ensemble Fusion", final_pred)
    ]
    
    summary = []
    for name, pred in models:
        summary.append({
            "MODEL": name,
            "ACCURACY": accuracy_score(y_test, pred),
            "PRECISION": precision_score(y_test, pred, zero_division=0),
            "RECALL": recall_score(y_test, pred, zero_division=0),
            "F1-SCORE": f1_score(y_test, pred, zero_division=0)
        })

    # Save CSV for Streamlit Tabs 1-3
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("results/latest_results.csv", index=False)
    
    # Save Comparison Bar Chart for Tab 4
    plot_final_comparison(summary)

    print("\n" + "─"*75)
    print(" FINAL PERFORMANCE SUMMARY ".center(75))
    print("─"*75)
    # Format decimals for terminal view
    display_df = summary_df.copy()
    for col in ["ACCURACY", "PRECISION", "RECALL", "F1-SCORE"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    print(display_df.to_string(index=False))
    print("─"*75)
    print("Pipeline Complete. Ready for Dashboard View.")

if __name__ == "__main__":
    main()