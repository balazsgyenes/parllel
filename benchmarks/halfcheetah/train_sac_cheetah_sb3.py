# fmt: off
from contextlib import contextmanager
from typing import Iterator

import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

import wandb


# fmt: on
def make_env() -> gym.Env:
    env = gym.make("HalfCheetah-v4")
    env = Monitor(env)  # record stats such as returns
    return env


@contextmanager
def build(config: DictConfig) -> Iterator[tuple[BaseAlgorithm, BaseCallback]]:
    with open_dict(config):
        config["algo"]["train_freq"] = max(
            1, config["algo"]["train_freq"] // config["n_parallel_envs"]
        )
        config["eval_interval_steps"] = max(
            1, config["eval_interval_steps"] // config["n_parallel_envs"]
        )

    env = DummyVecEnv([make_env for _ in range(config["n_parallel_envs"])])

    algo_config = OmegaConf.to_container(
        config["algo"],
        resolve=True,
        throw_on_missing=True,
    )
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"log_data/halfcheetah/{wandb.run.id}",
        **algo_config,
    )

    eval_env = DummyVecEnv([make_env for _ in range(config["n_parallel_envs"])])
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=config["eval_interval_steps"],
        deterministic=config["deterministic_eval_mode"],
    )

    callback = CallbackList(
        [
            eval_callback,
            WandbCallback(verbose=2),
        ]
    )

    try:
        yield model, callback
    finally:
        env.close()


@hydra.main(version_base=None, config_path="conf", config_name="sac_cheetah_sb3")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["continuous", "state-based", "sac", "feedforward", "cheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    with build(config) as (model, callback):
        model.learn(
            total_timesteps=int(config["n_steps"]),
            # log_interval is in episodes, and each episode is 1000 time steps
            log_interval=config["log_interval_steps"] // 1000,
            callback=callback,
        )

    run.finish()


if __name__ == "__main__":
    main()
