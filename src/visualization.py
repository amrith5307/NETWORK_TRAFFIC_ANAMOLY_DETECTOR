import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import os

def plot_final_comparison(summary_list, save_dir="results/plots"):
    """Creates a bar chart comparing Accuracy, Precision, Recall, and F1-Score."""
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame(summary_list)
    # Convert string percentages to floats for plotting
    for col in ["ACCURACY", "PRECISION", "RECALL", "F1-SCORE"]:
        df[col] = df[col].astype(float)

    df_melted = df.melt(id_vars="MODEL", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="MODEL", y="Score", hue="Metric", palette="magma")
    plt.title("Detailed Model Performance Comparison", fontsize=16)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "final_metrics_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison chart saved to {save_path}")

def plot_clusters(X, y, title="Network Traffic Clusters (PCA)", save_dir="results/plots"):
    """Reduces 42D data to 2D and plots with Outlier Clipping for better visuals."""
    os.makedirs(save_dir, exist_ok=True)
    print("Generating Optimized PCA Cluster Plot...")
    
    # --- FIX START ---
    # Ensure X is handled correctly regardless of type
    num_samples = X.shape[0]
    sample_size = min(5000, num_samples)
    indices = np.random.choice(num_samples, sample_size, replace=False)
    
    # If X is a numpy array (which it is after scaling), use this:
    if isinstance(X, np.ndarray):
        X_sample = X[indices]
    else:
        X_sample = X.iloc[indices].values

    # Handle y (the labels)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_sample = y.iloc[indices]
    else:
        y_sample = y[indices]
    # --- FIX END ---

    # 2. Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)

    # 3. ZOOM LOGIC: Clip the plot range to ignore extreme outliers
    x_min, x_max = np.percentile(X_pca[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_pca[:, 1], [1, 99])

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, 
                        cmap='coolwarm', alpha=0.6, s=15)
    
    plt.colorbar(scatter, label='Class (0: Normal, 1: Attack)')
    
    plt.xlim(x_min - 1, x_max + 1)
    plt.ylim(y_min - 1, y_max + 1)

    plt.title(title + " (Zoomed View)", fontsize=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = os.path.join(save_dir, "traffic_clusters_pca.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Improved PCA plot saved to {save_path}")