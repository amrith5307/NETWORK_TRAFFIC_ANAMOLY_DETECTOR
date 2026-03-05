import numpy as np
from sklearn.covariance import EllipticEnvelope

class HBOSDetector:
    """Histogram-based Outlier Score - Fast and effective for tabular data."""
    def __init__(self, n_bins=10, contamination=0.35):
        self.n_bins = n_bins
        self.contamination = contamination
        self.threshold = None

    def train(self, X):
        # We calculate histograms for each feature
        hist_data = []
        for i in range(X.shape[1]):
            hist, bin_edges = np.histogram(X[:, i], bins=self.n_bins, density=True)
            hist_data.append((hist, bin_edges))
        self.hist_data = hist_data

    def predict(self, X):
        # Simplified HBOS logic: lower density = higher anomaly score
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            hist, bin_edges = self.hist_data[i]
            # Find which bin each test point falls into
            bin_indices = np.digitize(X[:, i], bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
            # Use log density to prevent underflow
            scores += -np.log(hist[bin_indices] + 1e-10)
        
        self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return np.where(scores > self.threshold, 1, 0)

class EnvelopeDetector:
    """Elliptic Envelope - Better than OC-SVM for Gaussian-like distributions."""
    def __init__(self, contamination=0.35):
        self.model = EllipticEnvelope(contamination=contamination, random_state=42)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.where(y_pred == -1, 1, 0)