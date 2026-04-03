import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class IsolationForestModel:
    def __init__(self, n_estimators=200, contamination=0.46):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X):
        print("Training Isolation Forest...")
        self.model.fit(X)
        print("Isolation Forest training complete.")

    def predict(self, X):
        y_pred_raw = self.model.predict(X)
        return np.where(y_pred_raw == -1, 1, 0)

    def evaluate(self, X_test, y_test, save_dir="results/plots"):
        y_pred = self.predict(X_test)
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n--- Isolation Forest Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        plt.title("Isolation Forest Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"iso_forest_cm_{timestamp}.png"))
        plt.close()

        # CHANGED: overlay normal vs attack score distributions for better insight
        scores = self.model.decision_function(X_test)
        plt.figure(figsize=(6, 5))
        plt.hist(scores[np.array(y_test) == 0], bins=50, alpha=0.6,
                 color='steelblue', label='Normal')
        plt.hist(scores[np.array(y_test) == 1], bins=50, alpha=0.6,
                 color='tomato', label='Attack')
        plt.axvline(0, color='black', linestyle='--', label='Default threshold')
        plt.title("Isolation Forest Decision Scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"iso_forest_scores_{timestamp}.png"))
        plt.close()

        return y_pred