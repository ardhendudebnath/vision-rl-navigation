"""Training-time callbacks for navigation metrics and validation.

Episodic return is a poor progress signal for this task: reward shaping means
return can rise while success rate stays flat (the agent gets better at
approaching the goal without reaching it).  These callbacks log the metrics
the report is actually judged on, so training curves and results are the same
quantities.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from vision_nav.envs.nav_env import NavEnvConfig
from vision_nav.metrics.navigation import EpisodeResult, aggregate

__all__ = ["NavMetricsCallback", "ValidationCallback"]


class NavMetricsCallback(BaseCallback):
    """Log rolling success rate / SPL / collision rate over training episodes."""

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._episodes: deque[EpisodeResult] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Monitor stamps "episode" onto the info dict at episode end; it
            # is the reliable marker that this info is a terminal one.
            if "episode" not in info or "world_seed" not in info:
                continue
            self._episodes.append(EpisodeResult.from_info(info))

        if len(self._episodes) >= min(self.window, 20):
            m = aggregate(self._episodes)
            self.logger.record("nav/success_rate", m.success_rate)
            self.logger.record("nav/spl", m.spl)
            self.logger.record("nav/collision_rate", m.collision_rate)
            self.logger.record("nav/timeout_rate", m.timeout_rate)
            self.logger.record("nav/mean_steps_to_goal", m.mean_steps_to_goal)
        return True


class ValidationCallback(BaseCallback):
    """Periodically score the policy on the held-out validation split.

    Model selection uses **SPL on validation**, not training return.  Return
    is shaped and therefore not comparable across runs; SPL is the metric the
    final comparison reports, so selecting on it avoids picking a checkpoint
    that merely learned to farm the shaping term.
    """

    def __init__(
        self,
        env_config: NavEnvConfig,
        eval_freq: int = 25_000,
        n_episodes: int = 50,
        save_path: str | Path | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.env_config = env_config
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.save_path = Path(save_path) if save_path else None
        self.best_spl = -np.inf
        self.history: list[dict] = []
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._run_validation()
        return True

    def _on_training_end(self) -> None:
        self._run_validation()

    def _run_validation(self) -> None:
        from vision_nav.training.actors import SB3Actor
        from vision_nav.training.evaluate import evaluate

        actor = SB3Actor(self.model, deterministic=True)
        metrics, _ = evaluate(actor, self.env_config, n_episodes=self.n_episodes)

        for key, value in metrics.to_dict().items():
            if key != "n_episodes":
                self.logger.record(f"val/{key}", value)
        self.history.append({"timesteps": self.num_timesteps, **metrics.to_dict()})

        if self.verbose:
            print(
                f"[val @ {self.num_timesteps:>9,}] SR={metrics.success_rate:.3f} "
                f"SPL={metrics.spl:.3f} coll={metrics.collision_rate:.3f}",
                flush=True,
            )

        if metrics.spl > self.best_spl:
            self.best_spl = metrics.spl
            if self.save_path is not None:
                self.save_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"           new best SPL, saved {self.save_path}", flush=True)
