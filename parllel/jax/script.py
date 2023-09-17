import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import agent
import hydra
import jax
import jax.numpy as jnp
import optax
from envs.cartpole import build_cartpole
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
from models import ActorCriticModel
from omegaconf import DictConfig, OmegaConf
from runner import JaxRunner
from sampler import JaxSampler

import parllel.logger as logger
import wandb
from parllel import Array, ArrayDict, ArrayTree, dict_map
from parllel.cages import TrajInfo
from parllel.jax.ppo import PPO, build_loss_sample_tree
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
                              add_obs_normalization, add_reward_clipping,
                              add_reward_normalization,
                              build_cages_and_sample_tree)
from parllel.replays import BatchedDataLoader
from parllel.transforms import Compose
from parllel.types import BatchSpec


def add_agent_info(
    state,
    sample_tree: ArrayDict[Array],
    example_obs_batch: ArrayTree,
    key: jax.random.PRNGKey,
) -> ArrayDict[Array]:
    # get example output from agent
    _, agent_info = agent.step(state.apply_fn, state.params, example_obs_batch, key)

    batch_shape = sample_tree["done"].batch_shape

    # allocate array tree based on examples
    sample_tree["agent_info"] = dict_map(
        Array.from_numpy,
        agent_info[0],
        batch_shape=batch_shape,
    )

    return sample_tree


def create_train_state(
    params,
    model: nn.Module,
    learning_rate: float,
):
    tx = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state


@contextmanager
def build(config):
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["algo"]["discount"])
    parallel = config["parallel"]

    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=build_cartpole,
        env_kwargs={"max_episode_steps": 1000},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(action_space, spaces.Discrete)

    model = ActorCriticModel(
        actor_hidden_sizes=[256, 256],
        critic_hidden_sizes=[256, 256],
        action_dim=action_space.n,
    )

    init_shape = jnp.ones((1, *obs_space.shape))
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    initial_params = model.init(subkey, init_shape)["params"]

    state = create_train_state(initial_params, model, 0.001)

    key, subkey = jax.random.split(key)
    sample_tree = add_agent_info(state, sample_tree, metadata.example_obs_batch, subkey)
    sample_tree = add_bootstrap_value(sample_tree)

    # add several helpful transforms
    batch_transforms, step_transforms = [], []

    sample_tree, step_transforms = add_obs_normalization(
        sample_tree,
        step_transforms,
        initial_count=config["obs_norm_initial_count"],
    )

    sample_tree, batch_transforms = add_reward_normalization(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
    )

    sample_tree, batch_transforms = add_reward_clipping(
        sample_tree,
        batch_transforms,
        reward_clip_min=config["reward_clip_min"],
        reward_clip_max=config["reward_clip_max"],
    )

    # add advantage normalization, required for PPO
    sample_tree, batch_transforms = add_advantage_estimation(
        sample_tree,
        batch_transforms,
        discount=config["algo"]["discount"],
        gae_lambda=config["algo"]["gae_lambda"],
        normalize=config["algo"]["normalize_advantage"],
    )
    key, subkey = jax.random.split(key)
    sampler = JaxSampler(
        batch_spec,
        cages,
        agent,
        sample_tree,
        key=subkey,
        max_steps_decorrelate=10,
        obs_transform=Compose(step_transforms),  # type: ignore
        batch_transform=Compose(batch_transforms),  # type: ignore
    )

    loss_sample_tree = build_loss_sample_tree(sample_tree)
    loss_sample_tree = loss_sample_tree.to_ndarray()
    loss_sample_tree = loss_sample_tree.apply(jnp.asarray)


    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        n_batches=4,
        # pre_batches_transform=lambda tree: jax.device_put(tree.to_ndarray()),
    )
    algorithm = PPO(state=state, dataloader=dataloader, **config["algo"])

    runner = JaxRunner(sampler, algorithm, batch_spec, **config["runner"])

    try:
        yield runner, state

    finally:
        sampler.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="conf", config_name="train_ppo")
def main(config: DictConfig) -> None:

    # run = wandb.init(
    #     anonymous="must",  # for this example, send to wandb dummy account
    #     project="CartPole",
    #     tags=["discrete", "state-based", "ppo", "feedforward"],
    #     config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    #     sync_tensorboard=True,  # auto-upload any values logged to tensorboard
    #     save_code=True,  # save script used to start training, git commit, and patch
    #     mode="disabled",
    # )

    # logger.init(
    #     wandb_run=run,
    #     # this log_dir is used if wandb is disabled (using `wandb disabled`)
    #     tensorboard=True,
    #     output_files={
    #         "txt": "log.txt",
    #         # "csv": "progress.csv",
    #     },
    #     config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    #     model_save_path="model.pt",
    #     # verbosity=Verbosity.DEBUG,
    # )

    with build(config) as (runner, state):
        runner.run(state)

    run.finish()


if __name__ == "__main__":
    main()

