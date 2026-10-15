"""Nested Bernoulli model of two policies with known truth, to validate the toolkit.

Task t has a fixed success rate per policy. Each initial state of a task draws a success
probability from a Beta distribution around that rate, and each seed draws a Bernoulli outcome.
The variance between states of task t is proportional to p_t (1 - p_t), so that easy and hard
tasks stay inside (0, 1); `var_init` is its average over the tasks.
The two policies are tied at each level by a correlation: of their task rates, of their state
probabilities (the same quantile with probability `rho_init`) and of their seed noise (the same
uniform number with probability `rho_seed`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import stats

from evalstats.stats.variance import Components

# Points of the midpoint rule behind the true covariances.
_QUADRATURE_POINTS = 20_000


@dataclass(frozen=True)
class NestedBernoulli:
    """Two policies on `n_tasks` tasks. The defaults are one policy run twice on other seeds.

    For each policy, `p` is the success rate, `var_task` the variance between tasks and
    `var_init` the variance between initial states within a task. The variance between seeds is
    what is left of p(1 - p).
    """

    n_tasks: int = 10
    p: tuple[float, float] = (0.8, 0.8)
    var_task: tuple[float, float] = (0.02, 0.02)
    var_init: tuple[float, float] = (0.04, 0.04)
    rho_task: float = 1.0
    rho_init: float = 1.0
    rho_seed: float = 0.0

    def __post_init__(self) -> None:
        r = self.task_rates
        if r.min() <= 0 or r.max() >= 1:
            raise ValueError(f"task rates outside (0, 1): {r.round(3)}")
        if min(self.variance(0).seed, self.variance(1).seed) <= 0:
            raise ValueError("var_task + var_init must be below p (1 - p)")

    @cached_property
    def task_rates(self) -> np.ndarray:
        """Success rate of each policy on each task, shape (2, n_tasks)."""
        p, sd = np.array(self.p)[:, None], np.sqrt(self.var_task)[:, None]
        if not sd.any():
            return np.broadcast_to(p, (2, self.n_tasks)).copy()
        if self.n_tasks < 3:
            raise ValueError("a variance between tasks needs at least 3 tasks")
        # Two task profiles with mean 0, variance 1 and correlation 0 over the tasks.
        x = np.linspace(-1, 1, self.n_tasks)
        q, _ = np.linalg.qr(np.stack([np.ones_like(x), x, x**2], axis=1))
        z0, z1 = q[:, 1] * np.sqrt(self.n_tasks), q[:, 2] * np.sqrt(self.n_tasks)
        z = np.stack([z0, self.rho_task * z0 + np.sqrt(1 - self.rho_task**2) * z1])
        return p + sd * z

    def _beta(self, policy: int):
        """Distribution of the success probability of a state, one per task."""
        m = self.task_rates[policy][:, None]
        # Variance m (1 - m) / (k + 1) in each task, var_init on average over the tasks.
        k = (m * (1 - m)).mean() / self.var_init[policy] - 1
        return stats.beta(m * k, (1 - m) * k)

    def _state_prob(self, policy: int, u: np.ndarray) -> np.ndarray:
        """Success probability at quantile `u`, for `u` of shape (..., n_tasks, n_states)."""
        if self.var_init[policy] == 0:
            return np.broadcast_to(self.task_rates[policy][:, None], u.shape)
        return self._beta(policy).ppf(u)

    def _state_sf(self, policy: int, x: np.ndarray) -> np.ndarray:
        if self.var_init[policy] == 0:
            return (x < self.task_rates[policy][:, None]).astype(float)
        return self._beta(policy).sf(x)

    @cached_property
    def _state_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Per task, E[m0 m1] and E[min(m0, m1)] over the states, m the success probabilities."""
        u = ((np.arange(_QUADRATURE_POINTS) + 0.5) / _QUADRATURE_POINTS)[None, :]
        m0, m1 = self._state_prob(0, u), self._state_prob(1, u)
        r0, r1 = self.task_rates
        # Same quantile with probability rho_init, independent quantiles otherwise. For
        # independent variables on [0, 1], E[min] is the integral of the product of the
        # survival functions.
        independent_min = (self._state_sf(0, u) * self._state_sf(1, u)).mean(1)
        rho = self.rho_init
        prod = rho * (m0 * m1).mean(1) + (1 - rho) * r0 * r1
        low = rho * np.minimum(m0, m1).mean(1) + (1 - rho) * independent_min
        return prod, low

    def variance(self, policy: int) -> Components:
        """True variance components of one policy."""
        p, task, init = self.p[policy], self.var_task[policy], self.var_init[policy]
        return Components(task, init, p * (1 - p) - task - init)

    @cached_property
    def covariance(self) -> Components:
        """True components of the covariance between the two policies."""
        prod, low = self._state_moments
        r0, r1 = self.task_rates
        task = ((r0 - self.p[0]) * (r1 - self.p[1])).mean()
        init = (prod - r0 * r1).mean()
        # Given the two probabilities, the outcomes share their uniform number with probability
        # rho_seed, and then both succeed with probability min(m0, m1).
        seed = self.rho_seed * (low - prod).mean()
        return Components(float(task), float(init), float(seed))

    @cached_property
    def joint(self) -> np.ndarray:
        """Probabilities of (both succeed, only the first, only the second, both fail) on a key
        drawn at random: uniform task, new initial state, new seed."""
        both = self.covariance.total + self.p[0] * self.p[1]
        return np.array([both, self.p[0] - both, self.p[1] - both, 1 - sum(self.p) + both])

    def sample_grid(
        self, n_init: int, n_seeds: int, rng: np.random.Generator, size: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Outcomes of both policies on a grid, shape (2, *size, n_tasks, n_init, n_seeds)."""
        shape = (*size, self.n_tasks, n_init)
        u0 = rng.random(shape)
        u1 = np.where(rng.random(shape) < self.rho_init, u0, rng.random(shape))
        m0, m1 = self._state_prob(0, u0), self._state_prob(1, u1)
        w0 = rng.random((*shape, n_seeds))
        w1 = np.where(rng.random(w0.shape) < self.rho_seed, w0, rng.random(w0.shape))
        return np.stack([w0 < m0[..., None], w1 < m1[..., None]]).astype(np.int8)

    def sample_keys(self, shape: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Outcomes of both policies on independent random keys, shape (2, *shape).

        This is the infinite-population version of drawing keys from a grid: every key has a
        new initial state, so the pairs are independent with the probabilities of `joint`.
        """
        cell = rng.choice(4, size=shape, p=self.joint)
        return np.stack([cell <= 1, (cell == 0) | (cell == 2)]).astype(np.int8)
