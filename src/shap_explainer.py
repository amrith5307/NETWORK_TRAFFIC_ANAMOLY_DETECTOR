import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_analysis():
    """Generates Ma'am-approved SHAP graphs with Plain English labels."""
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.isolation_forest_model import IsolationForestModel

    # 1. Setup
    train_path = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_train.csv"
    test_path  = r"D:\network_traffic_anomaly_detector\data\raw\nsl_kdd_test.csv"
    plots_dir = os.path.join("results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Data & Model (Full Feature Space)
    train_raw = load_data(train_path)
    test_raw  = load_data(test_path)
    X_train, y_train, encoders = preprocess_data(train_raw, is_train=True)
    X_test, y_test = preprocess_data(test_raw, is_train=False, encoder_dict=encoders)

    iso = IsolationForestModel(n_estimators=200, contamination=0.46)
    iso.train(X_train)

    # 3. Prepare SHAP Data
    feature_names = encoders['feature_names']
    sample_data = shap.sample(X_test, 300, random_state=42)
    sample_df = pd.DataFrame(sample_data, columns=feature_names)

    # 4. Calculate SHAP
    explainer = shap.TreeExplainer(iso.model)
    shap_values = explainer(sample_df)

    # --- PLOT 1: GLOBAL BAR PLOT (Clean Labels) ---
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, show=False, max_display=15)
    
    # Change the math label on the X-axis to English
    plt.xlabel("Feature Importance Score (Impact on Model)", fontsize=12)
    plt.title("Top 15 Most Influential Network Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150)
    plt.close()

    # --- PLOT 2: WATERFALL PLOT (The "Math-to-English" Fix) ---
    plt.figure(figsize=(12, 8))
    # We use the first sample (index 0) as the case study
    shap.plots.waterfall(shap_values[0], show=False)
    
    # --- INTERNAL MAGIC: Rename the labels on the fly ---
    ax = plt.gca()
    # Find the text objects that SHAP puts on the chart
    for text in ax.texts:
        t = text.get_text()
        if "E[f(X)]" in t:
            # Replace base value label
            new_t = t.replace("E[f(X)]", "Average Baseline")
            text.set_text(new_t)
        if "f(x)" in t:
            # Replace final prediction label
            new_t = t.replace("f(x)", "Final Prediction")
            text.set_text(new_t)

    plt.title("Case Study: How the AI reached a Decision", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_waterfall.png"), dpi=150)
    plt.close()

    print(f"✅ SHAP plots generated with English labels in {plots_dir}")

if __name__ == "__main__":
    run_shap_analysis()