from contextlib import contextmanager
from typing import Iterator

import torch.nn
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback


@contextmanager
def build(config: DictConfig) -> Iterator[tuple[BaseAlgorithm, dict]]:
    with open_dict(config):
        # eval_interval_steps is defined in global steps, not steps per environment
        config["eval_interval_steps"] = max(
            1, config["eval_interval_steps"] // config["n_envs"]
        )

    env = make_vec_env(
        env_id=config["env_name"],
        n_envs=config["n_envs"],
    )
    if config["normalize"]:
        env = VecNormalize(env, gamma=config["algo"]["gamma"])

    algo_config = OmegaConf.to_container(
        config["algo"],
        resolve=True,
        throw_on_missing=True,
    )

    activation_fn = algo_config["policy_kwargs"]["activation_fn"]
    activation_fn = getattr(torch.nn, activation_fn)
    algo_config["policy_kwargs"]["activation_fn"] = activation_fn

    if isinstance(lr := algo_config["learning_rate"], str):
        schedule, initial_value = lr.split("_")
        assert schedule == "lin"
        initial_value = float(initial_value)

        def linear_schedule(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        algo_config["learning_rate"] = linear_schedule

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config["log_dir"],
        **algo_config,
    )

    eval_env = make_vec_env(
        env_id=config["env_name"],
        n_envs=config["n_eval_envs"],
    )
    if config["normalize"]:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_reward=False,
            gamma=config["algo"]["gamma"],
        )

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
