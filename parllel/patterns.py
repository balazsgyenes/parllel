# fmt: off
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np

import parllel.logger as logger
from parllel import Array, ArrayDict, ArrayOrMapping, ArrayTree, dict_map
from parllel.agents import Agent
from parllel.cages import Cage, ProcessCage, SerialCage
from parllel.samplers import EvalSampler
from parllel.transforms import (ClipRewards, Compose, EstimateAdvantage,
                                EstimateMultiAgentAdvantage,
                                NormalizeAdvantage, NormalizeObservations,
                                NormalizeRewards, StepTransform, Transform)
from parllel.types import BatchSpec

# fmt: on


@dataclass
class Metadata:
    obs_space: gym.Space
    action_space: gym.Space
    example_obs: ArrayOrMapping[np.ndarray]
    example_action: ArrayOrMapping[np.ndarray]
    example_reward: ArrayOrMapping[np.ndarray]
    example_info: ArrayOrMapping[np.ndarray]
    example_obs_batch: ArrayTree[np.ndarray] | None = None
    example_action_batch: ArrayTree[np.ndarray] | None = None


def build_cages_and_sample_tree(
    EnvClass: Callable,
    env_kwargs: Mapping[str, Any],
    TrajInfoClass: Callable,
    reset_automatically: bool,
    batch_spec: BatchSpec,
    parallel: bool,
    full_size: int | None = None,
    keys_to_skip: str | Sequence[str] = (),
) -> tuple[list[Cage], ArrayDict[Array], Metadata]:
    if parallel:
        CageCls = ProcessCage
        storage = "shared"
    else:
        CageCls = SerialCage
        storage = "local"

    if isinstance(keys_to_skip, str):
        keys_to_skip = (keys_to_skip,)

    cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfoClass,
        reset_automatically=reset_automatically,
    )

    # create example env
    example_cage = CageCls(**cage_kwargs)

    # get example output from env
    example_cage.random_step_async()
    (
        action,
        next_obs,
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = example_cage.await_step()

    example_cage.close()

    if full_size is not None:
        logger.debug(f"Allocating replay buffer of size {batch_spec.B * full_size}...")
    else:
        logger.debug("Allocating sample tree...")

    sample_tree: ArrayDict[Array] = ArrayDict()

    if {"obs", "observation"} & set(keys_to_skip) == set():
        # allocate sample tree based on examples
        logger.debug("Allocating next observations...")
        sample_tree["next_observation"] = dict_map(
            Array.from_numpy,
            next_obs,
            batch_shape=tuple(batch_spec),
            storage=storage,
            full_size=full_size,
        )
        logger.debug("Allocating observations...")
        sample_tree["observation"] = dict_map(
            Array.from_numpy,
            obs,
            batch_shape=tuple(batch_spec),
            storage=storage,
            padding=1,
            full_size=full_size,
        )

    if "reward" not in keys_to_skip:
        logger.debug("Allocating rewards...")
        # in case environment creates rewards of shape (1,) or of integer type,
        # force to be correct shape and type
        sample_tree["reward"] = dict_map(
            Array.from_numpy,
            reward,
            batch_shape=tuple(batch_spec),
            dtype=np.float32,
            feature_shape=(),
            storage=storage,
            full_size=full_size,
        )

    if "terminated" not in keys_to_skip:
        logger.debug("Allocating terminated...")
        sample_tree["terminated"] = Array.from_numpy(
            terminated,
            batch_shape=tuple(batch_spec),
            dtype=bool,
            feature_shape=(),
            storage=storage,
            full_size=full_size,  # used for SAC replay buffer
        )

    if "truncated" not in keys_to_skip:
        logger.debug("Allocating truncated...")
        sample_tree["truncated"] = Array.from_numpy(
            truncated,
            batch_shape=tuple(batch_spec),
            dtype=bool,
            feature_shape=(),
            storage=storage,
        )

    if "done" not in keys_to_skip:
        logger.debug("Allocating done...")
        # add padding in case reward normalization is used
        # TODO: ideally, we only would add padding if we know we want reward
        # normalization, but how to do this?
        sample_tree["done"] = Array.from_numpy(
            terminated,
            batch_shape=tuple(batch_spec),
            dtype=bool,
            feature_shape=(),
            storage=storage,
            padding=1,
        )

    if "env_info" not in keys_to_skip:
        logger.debug("Allocating env_info...")
        sample_tree["env_info"] = dict_map(
            Array.from_numpy,
            info,
            batch_shape=tuple(batch_spec),
            storage=storage,
        )

    if "action" not in keys_to_skip:
        logger.debug("Allocating actions...")
        # in discrete problems, integer actions are used as array indices during
        # optimization. Pytorch requires indices to be 64-bit integers, so we
        # force actions to be 32 bits only if they are floats
        sample_tree["action"] = dict_map(
            Array.from_numpy,
            action,
            batch_shape=tuple(batch_spec),
            force_32bit="float",
            storage=storage,
            full_size=full_size,
        )

    if "agent_info" not in keys_to_skip:
        logger.debug("Allocating agent_info...")
        # add empty agent_info field by default
        # user is free to set a different value later
        sample_tree["agent_info"] = ArrayDict()

    logger.info(f"Instantiating {batch_spec.B} environments...")

    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    logger.info("Environments instantiated.")

    # get obs and action spaces for metadata
    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    # get example obs and actions for metadata
    # write sample into the sample_tree and read it back out. This ensures the example is in a standard form (i.e. if using JaggedArray or LazyFramesArray)
    if "observation" in sample_tree:
        sample_tree["observation"][0] = obs_space.sample()
        obs_batch = sample_tree["observation"][0]
    else:
        obs_batch = None

    if "action" in sample_tree:
        sample_tree["action"][0] = action_space.sample()
        action_batch = sample_tree["action"][0]
    else:
        action_batch = None

    metadata = Metadata(
        obs_space=obs_space,
        action_space=action_space,
        example_obs_batch=obs_batch,
        example_action_batch=action_batch,
        example_obs=obs,
        example_action=action,
        example_reward=reward,
        example_info=info,
    )

    return cages, sample_tree, metadata


def add_agent_info(
    sample_tree: ArrayDict[Array],
    agent: Agent,
    example_obs_batch: ArrayTree,
) -> ArrayDict[Array]:
    # get example output from agent
    _, agent_info = agent.step(example_obs_batch)

    batch_shape = sample_tree["done"].batch_shape

    # allocate array tree based on examples
    sample_tree["agent_info"] = dict_map(
        Array.from_numpy,
        agent_info[0],
        batch_shape=batch_shape,
    )

    return sample_tree


def add_initial_rnn_state(
    sample_tree: ArrayDict[Array],
    agent: Agent,
) -> ArrayDict[Array]:
    rnn_state = agent.initial_rnn_state()
    sample_tree["initial_rnn_state"] = dict_map(
        Array.from_numpy,
        rnn_state[0],  # get rnn_state for a single environment
        batch_shape=sample_tree["done"].batch_shape[1:2],  # access batch_B
        storage="local",
    )
    return sample_tree


def add_bootstrap_value(sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
    # Create an array with only (B,) leading dimension
    # TODO: can we solve this by just adding padding to value?
    value = sample_tree["agent_info"]["value"]
    bootstrap_value = value[0].new_array(storage="local")
    sample_tree["bootstrap_value"] = bootstrap_value
    return sample_tree


def add_valid(sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
    # allocate new Array objects for valid
    sample_tree["valid"] = sample_tree["done"].new_array(storage="local")
    return sample_tree


def add_advantage_estimation(
    sample_tree: ArrayDict[Array],
    transforms: list[Transform],
    discount: float,
    gae_lambda: float,
    multiagent: bool = False,
    normalize: bool = False,
) -> tuple[ArrayDict[Array], list[Transform]]:
    # add required fields to sample_tree
    # get convenient local references
    value = sample_tree["agent_info"]["value"]

    if not isinstance(value, Array):
        raise TypeError("Value estimate must be a single tensor, not a dict.")

    if multiagent and not isinstance(sample_tree["action"], Mapping):
        raise TypeError("MultiAgent Advantage requires a dictionary action space.")

    value_shape = value.shape[value.n_batch_dims :]
    if multiagent and len(value_shape) == 0:
        # in algo, advantage must broadcast with distribution values (e.g.
        # log likelihood, likelihood ratio)
        advantage_shape = value_shape + (1,)
    else:
        advantage_shape = value_shape

    # allocate new Array objects for advantage and return_
    sample_tree["advantage"] = value.new_array(
        feature_shape=advantage_shape,
        storage="local",
    )
    # in algo, return_ must broadcast with value
    sample_tree["return_"] = value.new_array(storage="local")

    # create required transforms and add to list
    if multiagent:
        transforms.append(
            EstimateMultiAgentAdvantage(
                sample_tree=sample_tree,
                discount=discount,
                gae_lambda=gae_lambda,
            )
        )
    else:
        transforms.append(EstimateAdvantage(discount=discount, gae_lambda=gae_lambda))

    if normalize:
        transforms.append(NormalizeAdvantage(sample_tree=sample_tree))

    return sample_tree, transforms


def add_obs_normalization(
    sample_tree: ArrayDict[Array],
    transforms: list[Transform],
    initial_count: int | float | None = None,
) -> tuple[ArrayDict[Array], list[Transform]]:
    obs = sample_tree["observation"]
    if not isinstance(sample_tree["observation"], Array):
        raise NotImplementedError("Dictionary observations not supported.")
    # get feature shape of observation
    obs_shape = obs.shape[obs.n_batch_dims :]
    transforms.append(
        NormalizeObservations(
            sample_tree=sample_tree,
            obs_shape=obs_shape,
            initial_count=initial_count,
        )
    )
    return sample_tree, transforms


def add_reward_normalization(
    sample_tree: ArrayDict[Array],
    transforms: list[Transform],
    discount: float,
    initial_count: int | float | None = None,
) -> tuple[ArrayDict[Array], list[Transform]]:
    if sample_tree["done"].padding == 0:
        raise ValueError(
            "sample_tree['done'] must have padding >= 1 when using NormalizeRewards"
        )

    # TODO: handle multi-reward case

    # allocate new Array for past discounted returns
    sample_tree["past_return"] = sample_tree["reward"].new_array(
        padding=1,
        storage="local",
    )

    # create NormalizeReward transform and add to list
    transforms.append(
        NormalizeRewards(
            sample_tree=sample_tree,
            discount=discount,
            initial_count=initial_count,
        )
    )
    return sample_tree, transforms


def add_reward_clipping(
    sample_tree: ArrayDict[Array],
    transforms: list[Transform],
    reward_clip_min: float | None = None,
    reward_clip_max: float | None = None,
) -> tuple[ArrayDict[Array], list[Transform]]:
    transforms.append(
        ClipRewards(
            reward_min=reward_clip_min,
            reward_max=reward_clip_max,
        )
    )
    return sample_tree, transforms


def build_eval_sampler(
    sample_tree: ArrayDict[Array],
    agent: Agent,
    CageCls: type[Cage],
    EnvClass: Callable,
    env_kwargs: Mapping[str, Any],
    TrajInfoClass: Callable,
    n_eval_envs: int,
    max_traj_length: int,
    max_trajectories: int,
    step_transforms: list[StepTransform] | None = None,
) -> tuple[EvalSampler, ArrayDict[Array]]:
    logger.debug("Allocating eval sample tree...")

    # allocate a sample tree with space for a single time step
    # first, collect only the keys needed for evaluation
    eval_tree_keys = [
        "action",
        "observation",
        "terminated",
        "truncated",
        "done",
    ]
    eval_tree_example = ArrayDict(
        {key: sample_tree[key] for key in eval_tree_keys},
    )
    # create a new tree with leading dimensions (1, B_eval)
    eval_sample_tree = eval_tree_example.new_array(batch_shape=(1, n_eval_envs))

    eval_cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfoClass,
        reset_automatically=True,
    )
    eval_envs = [CageCls(**eval_cage_kwargs) for _ in range(n_eval_envs)]

    if step_transforms is not None:
        step_transforms = Compose(step_transforms)

    eval_sampler = EvalSampler(
        max_traj_length=max_traj_length,
        max_trajectories=max_trajectories,
        envs=eval_envs,
        agent=agent,
        sample_tree=eval_sample_tree,
        obs_transform=step_transforms,
    )

    return eval_sampler, eval_sample_tree
