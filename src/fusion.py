import numpy as np

class AnomalyFuser:
    def __init__(self, model_predictions):
        # We turn the list of predictions into a mathematical matrix
        self.predictions = np.array(model_predictions)

    def majority_vote(self):
        # If 2 or 3 models say '1', we confirm it's an attack
        vote_sum = np.sum(self.predictions, axis=0)
        return np.where(vote_sum >= 2, 1, 0)

    def unanimous_vote(self):
        # Extremely strict: all 3 must agree
        vote_sum = np.sum(self.predictions, axis=0)
        return np.where(vote_sum == 3, 1, 0)