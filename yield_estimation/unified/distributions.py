"""Dependency-light probability distributions for yield estimation."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union


def _as_2d(samples: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"expected samples with shape (N, {dimension}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("samples contain NaN or Inf")
    return values


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    safe_maximum = np.where(finite, maximum, 0.0)
    result = safe_maximum + np.log(np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True))
    result = np.where(finite, result, -np.inf)
    return np.squeeze(result, axis=axis)


class GaussianDistribution:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        dimension = self.mean.size
        if self.covariance.shape != (dimension, dimension):
            raise ValueError(f"covariance must have shape {(dimension, dimension)}")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise ValueError("mean/covariance contains NaN or Inf")
        self._chol = np.linalg.cholesky(self.covariance)
        self._log_det = 2.0 * np.log(np.diag(self._chol)).sum()

    @property
    def dimension(self) -> int:
        return self.mean.size

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return np.empty((0, self.dimension), dtype=np.float64)
        z = rng.normal(size=(count, self.dimension))
        return self.mean[None, :] + z @ self._chol.T

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        values = _as_2d(samples, self.dimension)
        solved = np.linalg.solve(self._chol, (values - self.mean[None, :]).T).T
        quadratic = np.sum(solved * solved, axis=1)
        normalizer = self.dimension * np.log(2.0 * np.pi) + self._log_det
        return -0.5 * (normalizer + quadratic)

    def standardize(self, samples: np.ndarray) -> np.ndarray:
        values = _as_2d(samples, self.dimension)
        return np.linalg.solve(self._chol, (values - self.mean[None, :]).T).T


class GaussianMixture:
    """Gaussian mixture with a shared diagonal covariance."""

    def __init__(
        self,
        centers: np.ndarray,
        variance: Union[float, np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        if self.centers.ndim != 2 or self.centers.shape[0] == 0:
            raise ValueError("centers must be a non-empty 2D array")
        self.dimension = self.centers.shape[1]
        var = np.asarray(variance, dtype=np.float64)
        if var.ndim == 0:
            var = np.full(self.dimension, float(var), dtype=np.float64)
        self.variance = var.reshape(-1)
        if self.variance.size != self.dimension or np.any(self.variance <= 0):
            raise ValueError("variance must be positive with one value per dimension")
        if weights is None:
            weights = np.ones(self.centers.shape[0], dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if self.weights.size != self.centers.shape[0] or np.any(self.weights < 0):
            raise ValueError("invalid mixture weights")
        self.weights /= self.weights.sum()
        self._log_weights = np.log(np.maximum(self.weights, np.finfo(float).tiny))
        self._log_norm = -0.5 * (
            self.dimension * np.log(2.0 * np.pi) + np.log(self.variance).sum()
        )

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count == 0:
            return np.empty((0, self.dimension), dtype=np.float64)
        indices = rng.choice(self.centers.shape[0], size=count, p=self.weights)
        noise = rng.normal(size=(count, self.dimension)) * np.sqrt(self.variance)[None, :]
        return self.centers[indices] + noise

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        values = _as_2d(samples, self.dimension)
        output = np.empty(values.shape[0], dtype=np.float64)
        chunk = 256
        for start in range(0, values.shape[0], chunk):
            stop = min(start + chunk, values.shape[0])
            diff = values[start:stop, None, :] - self.centers[None, :, :]
            quadratic = np.sum(diff * diff / self.variance[None, None, :], axis=2)
            output[start:stop] = _logsumexp(
                self._log_norm - 0.5 * quadratic + self._log_weights[None, :],
                axis=1,
            )
        return output


class LegacyDistributionAdapter:
    """Adapt OpenYield's historical ``generate/log_pdf`` distributions."""

    def __init__(self, distribution: object, mean: np.ndarray) -> None:
        self.distribution = distribution
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.dimension = self.mean.size

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if hasattr(self.distribution, "sample"):
            try:
                values = self.distribution.sample(count, rng)
            except TypeError:
                values = self.distribution.sample(count)
        elif hasattr(self.distribution, "generate"):
            state = np.random.get_state()
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            try:
                values = self.distribution.generate(count)
            finally:
                np.random.set_state(state)
        else:
            raise TypeError("legacy distribution needs sample() or generate()")
        return _as_2d(values, self.dimension)

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        values = self.distribution.log_pdf(_as_2d(samples, self.dimension))
        return np.asarray(values, dtype=np.float64).reshape(-1)
