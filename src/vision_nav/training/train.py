"""PPO training entry point (Hydra-configured).

Run with::

    python -m vision_nav.training.train
    python -m vision_nav.training.train algo.n_steps=1024 train.total_timesteps=2000000
    python -m vision_nav.training.train --config-name smoke

Every run writes its resolved config alongside its checkpoints, so a result
can always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from vision_nav.training.callbacks import NavMetricsCallback, ValidationCallback
from vision_nav.training.env_factory import build_env_config, make_vec_env

__all__ = ["train", "main"]

_CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> str:
    """Pick the training device.

    For a small MLP policy on a fast CPU-side simulator, CPU usually *beats*
    GPU: the per-batch host-device transfer costs more than the matmuls save.
    ``auto`` therefore prefers CPU for MLP policies, which is the opposite of
    the usual default and worth stating explicitly.
    """
    if requested != "auto":
        return requested
    return "cpu"


def _log_formats(requested) -> list[str]:
    """Drop log formats whose backend is not installed.

    TensorBoard is an optional extra; a missing optional dependency should not
    abort a training run that is otherwise ready to go.
    """
    formats = list(requested)
    if "tensorboard" in formats:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            formats.remove("tensorboard")
            print(
                "[warn] tensorboard not installed; logging CSV only. "
                "Install with: pip install tensorboard",
                flush=True,
            )
    return formats


def train(cfg: DictConfig) -> dict:
    """Run one training job. Returns the final validation metrics."""
    run_dir = Path(cfg.train.output_dir) / cfg.train.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    env_cfg_dict = OmegaConf.to_container(cfg.env, resolve=True)
    train_env_config = build_env_config(env_cfg_dict, split="train", n_worlds=cfg.train.n_worlds)
    val_env_config = build_env_config(env_cfg_dict, split="val", n_worlds=cfg.train.val_episodes)

    vec_env = make_vec_env(
        train_env_config,
        n_envs=cfg.train.n_envs,
        seed=cfg.train.seed,
        subprocess=cfg.train.subprocess,
    )

    logger = configure(str(run_dir / "logs"), _log_formats(cfg.train.log_formats))

    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    policy_kwargs = algo.pop("policy_kwargs", None) or {}
    if "net_arch" in policy_kwargs:
        policy_kwargs["net_arch"] = list(policy_kwargs["net_arch"])
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = getattr(torch.nn, policy_kwargs["activation_fn"])

    model = PPO(
        policy=algo.pop("policy", "MlpPolicy"),
        env=vec_env,
        seed=cfg.train.seed,
        device=device,
        policy_kwargs=policy_kwargs,
        **algo,
    )
    model.set_logger(logger)

    validation = ValidationCallback(
        val_env_config,
        eval_freq=cfg.train.eval_freq,
        n_episodes=cfg.train.val_episodes,
        save_path=run_dir / "best_model",
    )
    callbacks = CallbackList(
        [
            NavMetricsCallback(window=cfg.train.metrics_window),
            validation,
            CheckpointCallback(
                save_freq=max(cfg.train.checkpoint_freq // cfg.train.n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo",
            ),
        ]
    )

    print(
        f"\nTraining '{cfg.train.run_name}' | {cfg.train.total_timesteps:,} steps "
        f"| {cfg.train.n_envs} envs | device={device}\n",
        flush=True,
    )
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callbacks,
        progress_bar=cfg.train.progress_bar,
    )

    model.save(run_dir / "final_model")
    vec_env.close()

    history_path = run_dir / "validation_history.json"
    history_path.write_text(json.dumps(validation.history, indent=2), encoding="utf-8")

    summary = validation.history[-1] if validation.history else {}
    print(f"\nRun complete. Artefacts in {run_dir}")
    print(f"Best validation SPL: {validation.best_spl:.3f}")
    return summary


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
