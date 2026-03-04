import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class KMeansAnomalyDetector:
    def __init__(self, n_clusters=10, threshold_percentile=70, random_state=42): # Changed: clusters 5->10, percentile 95->70
        self.n_clusters = n_clusters
        self.threshold_percentile = threshold_percentile
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.threshold = None

    def train(self, X):
        print("Training KMeans...")
        self.model.fit(X)
        distances = np.min(self.model.transform(X), axis=1)
        # By setting this to 70, we say "If a point is in the furthest 30%, it's an attack"
        self.threshold = np.percentile(distances, self.threshold_percentile)
        print(f"KMeans training complete. Threshold: {self.threshold:.4f}")

    def predict(self, X):
        distances = np.min(self.model.transform(X), axis=1)
        return np.where(distances > self.threshold, 1, 0)

    def evaluate(self, X, y_true, save_dir="results/plots"):
        y_pred = self.predict(X)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n--- KMeans Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.title("KMeans Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"kmeans_cm_{timestamp}.png"))
        plt.close()

        return y_pred