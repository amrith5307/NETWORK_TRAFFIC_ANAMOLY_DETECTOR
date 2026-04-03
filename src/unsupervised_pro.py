import numpy as np
from sklearn.covariance import EllipticEnvelope


class EnvelopeDetector:
    def __init__(self, contamination=0.46):
        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )

    def train(self, X):
        print("Training Elliptic Envelope...")
        self.model.fit(X)
        print("Elliptic Envelope training complete.")

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.where(y_pred == -1, 1, 0)