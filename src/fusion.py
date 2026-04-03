import numpy as np
from sklearn.metrics import f1_score


class AnomalyFuser:
    def __init__(self, iso_model, env_model, iso_weight=0.4, env_weight=0.6):
        """
        CHANGED: Now takes the actual model objects instead of binary predictions.
        Uses raw decision scores instead of binary votes — much more precise.
        Envelope gets 60% weight since it consistently outperforms IsoForest.
        """
        self.iso_model = iso_model
        self.env_model = env_model
        self.iso_weight = iso_weight
        self.env_weight = env_weight
        self.threshold = 0.5  # will be tuned automatically

    def _blend_scores(self, X):
        """Weighted blend of normalized decision scores from both models."""
        iso_scores = self.iso_model.model.decision_function(X)
        env_scores = self.env_model.model.decision_function(X)

        # Normalize each to [0, 1] — makes weights meaningful across different scales
        def normalize(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx != mn else np.zeros_like(s)

        iso_norm = normalize(iso_scores)
        env_norm = normalize(env_scores)

        # Higher blended score = more normal, lower = more anomalous
        return self.iso_weight * iso_norm + self.env_weight * env_norm

    def tune_threshold(self, X_val, y_val):
        """
        NEW: Sweeps 200 thresholds on validation data and picks the one
        with the best F1. Removes guesswork from threshold selection.
        """
        blended = self._blend_scores(X_val)
        best_f1 = -1
        best_thresh = 0.5

        for t in np.linspace(0.01, 0.99, 200):
            preds = (blended < t).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = t

        self.threshold = best_thresh
        print(f"Best threshold: {best_thresh:.4f} → F1: {best_f1:.4f}")
        return best_thresh

    def predict(self, X):
        blended = self._blend_scores(X)
        return (blended < self.threshold).astype(int)

    def two_model_vote(self, predictions):
        """
        Kept for fallback — majority vote on binary predictions.
        predictions: [iso_pred, env_pred]
        """
        arr = np.array(predictions)
        votes = np.sum(arr, axis=0)
        return (votes >= 1).astype(int)