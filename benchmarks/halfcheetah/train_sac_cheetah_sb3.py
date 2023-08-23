import multiprocessing as mp
from contextlib import contextmanager
from functools import partial

import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

import wandb


@contextmanager
def build(config: DictConfig) -> BaseAlgorithm:
    build_env = partial(gym.make, "HalfCheetah-v4")

    algo_config = OmegaConf.to_container(
        config["algo"],
        resolve=True,
        throw_on_missing=True,
    )
    model = SAC("MlpPolicy", build_env(), verbose=1, **algo_config)

    try:
        yield model
    finally:
        pass


@hydra.main(version_base=None, config_path="conf", config_name="sac_cheetah_sb3")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["continuous", "state-based", "sac", "feedforward", "sb3", "cheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    with build(config) as model:
        model.learn(total_timesteps=int(config["n_steps"]), log_interval=4)

    run.finish()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
