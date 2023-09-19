from contextlib import contextmanager
from typing import Iterator

import hydra
import torch.nn
import wandb
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


@hydra.main(version_base=None, config_path="conf")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["ppo", config["env_name"]],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    with open_dict(config):
        config["log_dir"] = run.dir  # algo needs to know where to save tensorboard logs

    with build(config) as (model, learn_kwargs):
        model.learn(**learn_kwargs)

    run.finish()


if __name__ == "__main__":
    main()
