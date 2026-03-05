import numpy as np

class AnomalyFuser:
    def __init__(self, predictions):
        """
        predictions: A list of numpy arrays [iso_pred, hbos_pred, env_pred]
        Each array contains 0 (Normal) or 1 (Attack)
        """
        self.predictions = np.array(predictions)

    def majority_vote(self):
        """Standard 2-out-of-3 voting."""
        # Sum the columns: if sum is 2 or 3, it's an attack
        votes = np.sum(self.predictions, axis=0)
        return (votes >= 2).astype(int)

    def weighted_vote(self):
        """
        Smart voting: Envelope and IsoForest are much more accurate 
        than HBOS, so we give them more power.
        """
        # iso_weight=1.0, hbos_weight=0.5, env_weight=1.5
        weights = np.array([1.0, 0.5, 1.5]).reshape(-1, 1)
        
        # Calculate weighted sum
        weighted_sums = np.sum(self.predictions * weights, axis=0)
        
        # Threshold: If total weight > 1.5, classify as Attack
        return (weighted_sums > 1.5).astype(int)