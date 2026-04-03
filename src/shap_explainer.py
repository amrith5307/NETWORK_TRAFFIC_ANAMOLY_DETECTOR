import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_analysis():
    """Runs SHAP on IsolationForest and maps PCA back to original feature names."""
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.isolation_forest_model import IsolationForestModel

    # Paths (Keeping your D: drive paths)
    train_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_train.csv"
    test_path  = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_test.csv"

    # 1. Load Data
    train_raw = load_data(train_path)
    test_raw  = load_data(test_path)

    # 2. Extract Original Feature Names (Before PCA)
    # This ensures your graph labels are human-readable
    temp_df = train_raw.iloc[:, :41].copy() # Get feature columns
    # Replicate the drop logic from your preprocessing.py
    COLS_TO_DROP = ['num_outbound_cmds', 'is_host_login']
    original_cols = [c for c in temp_df.columns if c not in COLS_TO_DROP]

    # 3. Preprocess
    X_train_pca, y_train, encoders = preprocess_data(train_raw, is_train=True)
    X_test_pca, y_test = preprocess_data(test_raw, is_train=False, encoder_dict=encoders)

    # 4. Train Model
    iso = IsolationForestModel(n_estimators=200, contamination=0.46)
    iso.train(X_train_pca)

    # 5. PCA Mapping Logic
    # We find which original feature contributes most to each PCA component
    pca = encoders['pca']
    mapped_feature_names = []
    for i in range(pca.n_components_):
        # Find index of original feature with highest loading for this PC
        best_feature_idx = np.argmax(np.abs(pca.components_[i]))
        mapped_feature_names.append(f"{original_cols[best_feature_idx]} (PC{i+1})")

    # 6. Create a DataFrame for SHAP so it has names
    sample_pca = shap.sample(X_test_pca, 200, random_state=42)
    sample_df = pd.DataFrame(sample_pca, columns=mapped_feature_names)

    # 7. Explain Model
    # Isolation Forest works best with TreeExplainer
    explainer = shap.TreeExplainer(iso.model)
    shap_values = explainer.shap_values(sample_df)

    # 8. Plotting and Saving
    plots_dir = os.path.join("results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Summary/Beeswarm (The main one your app.py looks for)
    plt.figure(figsize=(10, 6))
    # We use 'shap_summary.png' because that's what your app.py expects!
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.title("Top Features Driving Anomaly Detection")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=100)
    plt.close()

    # Plot 2: Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar.png"), dpi=100)
    plt.close()

    print(f"SHAP Analysis complete. Plots saved to {plots_dir}")

if __name__ == "__main__":
    run_shap_analysis()