import numpy as np
from sklearn.base import BaseEstimator
from dataclasses import dataclass


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5
    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class WeightedMDS(BaseEstimator):
    """
    Implementation of weighted multidimensional scaling to dimension 1
    """
    def __init__(
        self,
        max_iter=300,
        weights=None,
        lr=1e-2,
        verbose=0
    ):
        assert np.allclose(weights.T, weights), "Weights matrix should be symmetrical"
        assert np.all(weights >= 0), "Weights should be non-negative"
        self.weights = weights
        self.max_iter = max_iter

        self.lr = LearningRate(lambda_=lr)
        self.alpha = 0.9
        self.h = np.zeros(self.weights.shape[0])

    def fit(self, dissimilarities):
        self.dissimilarities = dissimilarities
        if self.weights is None:
            self.weights = np.ones(self.dissimilarities.shape)
        return self

    def _compute_gradient(self, r):
        n_samples = self.dissimilarities.shape[0]
        grad = np.zeros(n_samples)
        # could be optimized
        for k in range(1, n_samples):
            grad[k] = 0.0
            for i in range(n_samples):
                d_dk = 2 * self.weights[i][k] * (np.abs(r[k] - r[i]) - self.dissimilarities[i][k])
                if r[k] < r[j]:
                    d_dk *= -1
                if i != k:
                    d_dk *= 2
                grad[k] += d_dk
        return grad

    def predict(self):
        n_samples = self.dissimilarities.shape[0]
        r = np.random.rand(n_samples)
        for _ in range(self.max_iter):
            self.h = self.alpha * self.h + self.lr() * self._compute_gradient(r)
            r -= self.h
        r -= r.min()
        return r