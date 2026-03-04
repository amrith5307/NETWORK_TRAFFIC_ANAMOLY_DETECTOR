import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

class DBSCANAnomalyDetector:
    def __init__(self, eps=0.5, min_samples=5):
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def predict(self, X):
        labels = self.model.fit_predict(X)
        return np.where(labels == -1, 1, 0)

    def evaluate(self, X, y_true, save_dir="results/plots"):
        y_pred = self.predict(X)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.title("DBSCAN Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"dbscan_cm_{timestamp}.png"))
        plt.close()
        
        return y_pred