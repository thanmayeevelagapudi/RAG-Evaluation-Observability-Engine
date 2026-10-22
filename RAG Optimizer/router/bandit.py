"""Epsilon-greedy multi-armed bandit for strategy selection.

Pure Python — no async, no DB. Takes scores from the registry and returns
a strategy name + whether the choice was exploratory.
"""

from __future__ import annotations

import random
from typing import NamedTuple

from config.settings import get_settings


class Selection(NamedTuple):
    strategy: str
    was_exploration: bool


class EpsilonGreedyBandit:
    """Select strategies using epsilon-greedy exploration/exploitation.

    With probability epsilon, explore (pick any strategy at random).
    Otherwise exploit (pick the highest-scoring strategy).
    Any strategy with fewer than min_samples observations is always explored first.
    """

    def __init__(
        self,
        epsilon: float | None = None,
        min_samples: int | None = None,
    ) -> None:
        settings = get_settings()
        self.epsilon = epsilon if epsilon is not None else settings.bandit_epsilon
        self.min_samples = (
            min_samples if min_samples is not None else settings.bandit_min_samples
        )

    def select(
        self,
        scores: dict[str, float],
        sample_counts: dict[str, int],
    ) -> Selection:
        """Return the selected strategy name and whether it was an exploration.

        Args:
            scores:        {strategy: avg_score} from the registry.
            sample_counts: {strategy: n_observations} from the registry.
        """
        if not scores:
            raise ValueError("scores dict is empty — no strategies registered")

        # Force exploration for any strategy lacking sufficient data
        cold = [s for s, c in sample_counts.items() if c < self.min_samples]
        if cold:
            return Selection(strategy=random.choice(cold), was_exploration=True)

        best = max(scores, key=lambda s: scores[s])

        if random.random() < self.epsilon:
            # Exploration: pick any strategy at random (was_exploration always True)
            return Selection(strategy=random.choice(list(scores.keys())), was_exploration=True)

        return Selection(strategy=best, was_exploration=False)
