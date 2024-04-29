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
    Weighted multidimensional scaling (to dimension 1).
    """
    def __init__(
        self,
        max_iter=300,
        weights=None,
        lr=1e-2,
        verbose=0
    ):
        """
        Parameters
        ----------
        max_iter : int, default=300
            Maximum number of iterations of the weighted MDS algorithm.

        weights : ndarray of shape (n_samples, n_samples)
            Weights for wMDS.
        
        lr : float, default=1e-2
            Learning rate for wMDS gradient descent.
        """
        assert np.allclose(weights.T, weights), "Weights matrix should be symmetrical"
        assert np.all(weights >= 0), "Weights should be non-negative"
        self.weights = weights
        self.max_iter = max_iter

        self.lr = LearningRate(lambda_=lr)
        self.alpha = 0.9
        self.h = np.zeros(self.weights.shape[0])

    def fit(self, X, y=None, init=None):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples)
            Dissimilarity matrix. 

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples,), default=None
            Starting configuration of the `embedding` to initialize the wMDS
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.dissimilarities = X
        if self.weights is None:
            self.weights = np.ones(self.dissimilarities.shape)
        return self

    def _compute_gradient(self, r):
        """
        Computes gradient of loss function with respect to r.

        Parameters
        ----------
        r : ndarray of shape (n_samples,)
            The vector in which the gradient must be calculated.

        Returns:
        -------
        grad : ndarray of shape (n_samples,)
            Gradient of loss with respect to r.
        """
        n_samples = self.dissimilarities.shape[0]
        grad = np.zeros(n_samples)
        # Computations can be vectorized.
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
        """
        Returns:
        -------
        r : ndarray of shape (n_samples,)
            Result of wMDS algorithm.
        """
        n_samples = self.dissimilarities.shape[0]
        r = np.random.rand(n_samples)
        # Gradient descent with momentum
        for _ in range(self.max_iter):
            self.h = self.alpha * self.h + self.lr() * self._compute_gradient(r)
            r -= self.h
        r -= r.min()
        return r
