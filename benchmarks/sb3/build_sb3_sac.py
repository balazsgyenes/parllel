from contextlib import contextmanager
from typing import Iterator

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback


@contextmanager
def build(config: DictConfig) -> Iterator[tuple[BaseAlgorithm, BaseCallback]]:
    with open_dict(config):
        config["algo"]["train_freq"] = max(
            1, config["algo"]["train_freq"] // config["n_envs"]
        )
        config["eval_interval_steps"] = max(
            1, config["eval_interval_steps"] // config["n_envs"]
        )

    def make_env() -> gym.Env:
        env = gym.make(config["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env for _ in range(config["n_envs"])])

    algo_config = OmegaConf.to_container(
        config["algo"],
        resolve=True,
        throw_on_missing=True,
    )

    if isinstance(lr := algo_config["learning_rate"], str):
        schedule, initial_value = lr.split("_")
        assert schedule == "lin"
        initial_value = float(initial_value)

        def linear_schedule(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        algo_config["learning_rate"] = linear_schedule

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config["log_dir"],
        **algo_config,
    )

    eval_env = DummyVecEnv([make_env for _ in range(config["n_envs"])])
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=config["n_eval_episodes"],
        eval_freq=config["eval_interval_steps"],
        deterministic=config["deterministic_eval_mode"],
    )

    learn_kwargs = {
        "callback": CallbackList(
            [
                eval_callback,
                WandbCallback(verbose=2),
            ]
        ),
        "total_timesteps": int(config["n_steps"]),
        "log_interval": config["log_interval_episodes"],
    }

    try:
        yield model, learn_kwargs
    finally:
        env.close()
