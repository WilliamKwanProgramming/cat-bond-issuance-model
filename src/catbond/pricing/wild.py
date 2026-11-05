"""Wild cluster bootstrap test and confidence interval for one OLS coefficient.

Restricted wild cluster bootstrap-t ("WCR", Cameron, Gelbach and Miller 2008), with the
interval obtained by inverting the test (Roodman, MacKinnon, Nielsen and Webb 2019):

For a null value beta_0 of coefficient j:

1. restricted fit: regress y - beta_0 x_j on the other columns X_(-j); fitted values
   yhat_0 = X_(-j) gamma + beta_0 x_j, residuals u = y - yhat_0;
2. for b = 1..B draw one weight w_bg per cluster g (Rademacher: +1 or -1 with probability 1/2)
   and set y*_b = yhat_0 + w_b,g(i) u_i;
3. refit the full model on y*_b and compute t*_b = (b*_bj - beta_0) / se*_bj with the CR1
   cluster-robust standard error (same formula and small-sample factor as the original fit);
4. p(beta_0) = (1/B) #{ |t*_b| >= |t| }, where t = (bhat_j - beta_0) / se_j from the data.

The 1 - alpha interval is {beta_0 : p(beta_0) > alpha}. Its end points are found by stepping out
from bhat_j in units of the cluster SE until p <= alpha and then bisecting; the same B weight
vectors are used for every beta_0, so p is a deterministic function of beta_0 for a given seed.
p is a step function, so the end points are located to within ``tol`` times the cluster SE.

The CR1 standard error of coefficient j for residual vector e is
sqrt(c sum_g (sum_{i in g} q_i e_i)^2) with q = X (X'X)^-1 e_j and c = G/(G-1) (n-1)/(n-k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rademacher(rng: np.random.Generator, n_boot: int, n_clusters: int) -> np.ndarray:
    """(n_boot, n_clusters) matrix of independent +1/-1 weights with probability 1/2 each."""
    return 2.0 * rng.integers(0, 2, size=(n_boot, n_clusters)) - 1.0


@dataclass(frozen=True)
class WildResult:
    """Wild cluster bootstrap results for one coefficient."""

    estimate: float
    se_cluster: float
    lo: float
    hi: float
    level: float
    p_value_at: dict[float, float]
    n_boot: int
    n_clusters: int


class WildClusterBootstrap:
    """Restricted wild cluster bootstrap-t for coefficient ``j`` of the regression of y on x."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        j: int,
        n_boot: int,
        rng: np.random.Generator,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.j = j
        n, k = self.x.shape
        self.n_clusters = int(self.groups.max()) + 1
        if self.n_clusters < 2:
            raise ValueError("need at least two clusters")
        self.xtx_inv = np.linalg.inv(self.x.T @ self.x)
        self.q = self.x @ self.xtx_inv[:, j]
        self.c = self.n_clusters / (self.n_clusters - 1) * (n - 1) / (n - k)
        self.member = np.zeros((self.n_clusters, n))
        self.member[self.groups, np.arange(n)] = 1.0
        self.weights = rademacher(rng, n_boot, self.n_clusters)
        self.x_other = np.delete(self.x, j, axis=1)
        self.estimate = float(self.xtx_inv[j] @ self.x.T @ self.y)
        self.se = float(self._se(self.y - self.x @ (self.xtx_inv @ self.x.T @ self.y))[0])

    def _se(self, e: np.ndarray) -> np.ndarray:
        """CR1 SE of coefficient j for residual vectors in the columns of ``e`` (n or n x B)."""
        e2 = e.reshape(len(self.q), -1)
        s = self.member @ (self.q[:, None] * e2)
        return np.sqrt(self.c * np.sum(s**2, axis=0))

    def p_value(self, beta0: float) -> float:
        """Bootstrap p-value of H0: coefficient j = beta0 (two-sided, symmetric)."""
        xj = self.x[:, self.j]
        yr = self.y - beta0 * xj
        gamma, *_ = np.linalg.lstsq(self.x_other, yr, rcond=None)
        fitted0 = self.x_other @ gamma + beta0 * xj
        u = self.y - fitted0
        ystar = fitted0[:, None] + u[:, None] * self.weights[:, self.groups].T
        bstar = self.xtx_inv @ self.x.T @ ystar
        estar = ystar - self.x @ bstar
        tstar = (bstar[self.j] - beta0) / self._se(estar)
        t = (self.estimate - beta0) / self.se
        return float(np.mean(np.abs(tstar) >= abs(t) - 1e-12))

    def _endpoint(self, direction: float, alpha: float, tol: float, max_steps: int) -> float:
        inside, step = self.estimate, self.se
        outside = None
        for _ in range(max_steps):
            cand = inside + direction * step
            if self.p_value(cand) <= alpha:
                outside = cand
                break
            inside = cand
        if outside is None:
            return float(direction * np.inf)
        while abs(outside - inside) > tol * self.se:
            mid = 0.5 * (inside + outside)
            if self.p_value(mid) > alpha:
                inside = mid
            else:
                outside = mid
        return float(0.5 * (inside + outside))

    def interval(
        self,
        level: float = 0.95,
        null_values: tuple[float, ...] = (),
        tol: float = 1e-4,
        max_steps: int = 60,
    ) -> WildResult:
        """Test-inversion interval at ``level`` and p-values at the ``null_values``."""
        alpha = 1.0 - level
        lo = self._endpoint(-1.0, alpha, tol, max_steps)
        hi = self._endpoint(1.0, alpha, tol, max_steps)
        return WildResult(
            estimate=self.estimate,
            se_cluster=self.se,
            lo=lo,
            hi=hi,
            level=level,
            p_value_at={float(v): self.p_value(float(v)) for v in null_values},
            n_boot=self.weights.shape[0],
            n_clusters=self.n_clusters,
        )
