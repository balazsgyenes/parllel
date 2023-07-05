from collections.abc import MutableMapping
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import numpy as np

import parllel.logger as logger
from parllel import Array, ArrayDict
from parllel.cages import Cage, ProcessCage, SerialCage
from parllel.dict import dict_map
from parllel.handlers import Agent
from parllel.samplers import EvalSampler
from parllel.transforms import (ClipRewards, Compose, EstimateAdvantage,
                                EstimateMultiAgentAdvantage,
                                NormalizeAdvantage, NormalizeObservations,
                                NormalizeRewards, StepTransform, Transform)
from parllel.types import BatchSpec


def build_cages_and_env_buffers(
    EnvClass: Callable,
    env_kwargs: dict[str, Any],
    TrajInfoClass: Callable,
    reset_automatically: bool,
    batch_spec: BatchSpec,
    parallel: bool,
    full_size: Optional[int] = None,
) -> tuple[Sequence[Cage], ArrayDict[Array]]:
    if parallel:
        CageCls = ProcessCage
        storage = "shared"
    else:
        CageCls = SerialCage
        storage = "local"

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
    action, obs, reward, terminated, truncated, info = example_cage.await_step()

    example_cage.close()

    if full_size is not None:
        logger.debug(f"Allocating replay buffer of size {batch_spec.B * full_size}")
    else:
        logger.debug("Allocating batch buffer.")

    batch_buffer: ArrayDict[Array] = ArrayDict({})

    # allocate batch buffer based on examples
    batch_buffer["observation"] = dict_map(
        Array.from_numpy,
        obs,
        batch_shape=tuple(batch_spec),
        storage=storage,
        padding=1,
        full_size=full_size,
    )

    # in case environment creates rewards of shape (1,) or of integer type,
    # force to be correct shape and type
    batch_buffer["reward"] = dict_map(
        Array.from_numpy,
        reward,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=np.float32,
        storage=storage,
        full_size=full_size,
    )

    batch_buffer["terminated"] = Array.from_numpy(
        terminated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
        storage=storage,
    )

    batch_buffer["truncated"] = Array.from_numpy(
        truncated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
        storage=storage,
    )

    # add padding in case reward normalization is used
    # TODO: ideally, we only would add padding if we know we want reward
    # normalization, but how to do this?
    batch_buffer["done"] = Array.from_numpy(
        truncated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
        storage=storage,
        padding=1,
        full_size=full_size,
    )

    batch_buffer["env_info"] = dict_map(
        Array.from_numpy,
        info,
        batch_shape=tuple(batch_spec),
        storage=storage,
    )

    # in discrete problems, integer actions are used as array indices during
    # optimization. Pytorch requires indices to be 64-bit integers, so we
    # force actions to be 32 bits only if they are floats
    batch_buffer["action"] = dict_map(
        Array.from_numpy,
        action,
        batch_shape=tuple(batch_spec),
        force_32bit="float",
        storage=storage,
        full_size=full_size,
    )

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = batch_buffer

    logger.debug(f"Instantiating {batch_spec.B} environments...")

    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    logger.debug("Environments instantiated.")

    return cages, batch_buffer


def add_initial_rnn_state(
    batch_buffer: ArrayDict[Array],
    agent: Agent,
) -> ArrayDict[Array]:
    rnn_state = agent.initial_rnn_state()
    batch_buffer["init_rnn_state"] = dict_map(
        Array.from_numpy,
        rnn_state[0],  # get rnn_state for a single environment
        batch_shape=batch_buffer["done"].batch_shape[1:2],  # access batch_B
        storage="local",
    )
    return batch_buffer


def add_bootstrap_value(batch_buffer: ArrayDict[Array]) -> ArrayDict[Array]:
    # Create an array with only (B,) leading dimension
    # TODO: can we solve this by just adding padding to value?
    value = batch_buffer["agent_info"]["value"]
    bootstrap_value = value[0].new_array(storage="local")
    batch_buffer["bootstrap_value"] = bootstrap_value
    return batch_buffer


def add_valid(batch_buffer: ArrayDict[Array]) -> ArrayDict[Array]:
    # allocate new Array objects for valid
    batch_buffer["valid"] = batch_buffer["done"].new_array(storage="local")
    return batch_buffer


def add_advantage_estimation(
    batch_buffer: ArrayDict[Array],
    transforms: List[Transform],
    discount: float,
    gae_lambda: float,
    multiagent: bool = False,
    normalize: bool = False,
) -> tuple[ArrayDict[Array], List[Transform]]:
    # add required fields to batch_buffer
    # get convenient local references
    value = batch_buffer["agent_info"]["value"]

    if not isinstance(value, Array):
        raise TypeError("Value estimate must be a single tensor, not a dict.")

    if multiagent and not isinstance(batch_buffer["action"], MutableMapping):
        raise TypeError("MultiAgent Advantage requires a dictionary action space.")

    value_shape = value.shape[value.n_batch_dims :]
    if multiagent and len(value_shape) == 0:
        # in algo, advantage must broadcast with distribution values (e.g.
        # log likelihood, likelihood ratio)
        advantage_shape = value_shape + (1,)
    else:
        advantage_shape = value_shape

    # allocate new Array objects for advantage and return_
    batch_buffer["advantage"] = value.new_array(
        feature_shape=advantage_shape,
        storage="local",
    )
    # in algo, return_ must broadcast with value
    batch_buffer["return_"] = value.new_array(storage="local")

    # create required transforms and add to list
    if multiagent:
        transforms.append(
            EstimateMultiAgentAdvantage(
                batch_buffer=batch_buffer,
                discount=discount,
                gae_lambda=gae_lambda,
            )
        )
    else:
        transforms.append(EstimateAdvantage(discount=discount, gae_lambda=gae_lambda))

    if normalize:
        transforms.append(NormalizeAdvantage(batch_buffer=batch_buffer))

    return batch_buffer, transforms


def add_obs_normalization(
    batch_buffer: ArrayDict[Array],
    transforms: List[Transform],
    initial_count: Union[int, float, None] = None,
) -> tuple[ArrayDict[Array], List[Transform]]:
    transforms.append(
        NormalizeObservations(
            batch_buffer=batch_buffer,
            # get shape of observation assuming 2 leading dimensions
            obs_shape=batch_buffer["observation"].shape[2:],
            initial_count=initial_count,
        )
    )
    return batch_buffer, transforms


def add_reward_normalization(
    batch_buffer: ArrayDict[Array],
    transforms: List[Transform],
    discount: float,
    initial_count: Union[int, float, None] = None,
) -> tuple[ArrayDict[Array], List[Transform]]:
    if batch_buffer["done"].padding == 0:
        raise TypeError(
            "batch_buffer.env.done must have padding >= 1 when using "
            "NormalizeRewards"
        )

    # allocate new Array for past discounted returns
    batch_buffer["past_return"] = batch_buffer["reward"].new_array(
        padding=1,
        storage="local",
    )

    # create NormalizeReward transform and add to list
    transforms.append(
        NormalizeRewards(
            batch_buffer=batch_buffer,
            discount=discount,
            initial_count=initial_count,
        )
    )
    return batch_buffer, transforms


def add_reward_clipping(
    batch_buffer: ArrayDict[Array],
    transforms: List[Transform],
    reward_clip_min: Optional[float] = None,
    reward_clip_max: Optional[float] = None,
) -> tuple[ArrayDict[Array], List[Transform]]:
    transforms.append(
        ClipRewards(
            reward_min=reward_clip_min,
            reward_max=reward_clip_max,
        )
    )
    return batch_buffer, transforms


def build_eval_sampler(
    samples_buffer: ArrayDict[Array],
    agent: Agent,
    CageCls: Type[Cage],
    EnvClass: Callable,
    env_kwargs: dict[str, Any],
    TrajInfoClass: Callable,
    n_eval_envs: int,
    max_traj_length: int,
    min_trajectories: int,
    step_transforms: Optional[List[StepTransform]] = None,
) -> tuple[EvalSampler, ArrayDict[Array]]:
    # allocate a step buffer with space for a single time step
    # first, collect only the keys needed for evaluation
    step_buffer_keys = [
        "action",
        "agent_info",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "done",
        "env_info",
    ]
    step_buffer_example = ArrayDict(
        {key: samples_buffer[key] for key in step_buffer_keys},
    )
    # create a new buffer with leading dimensions (1, B_eval)
    step_buffer = step_buffer_example.new_array(batch_shape=(1, n_eval_envs))

    eval_cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs=env_kwargs,
        TrajInfoClass=TrajInfoClass,
        reset_automatically=True,
    )
    if issubclass(CageCls, ProcessCage):
        eval_cage_kwargs["buffers"] = step_buffer
    eval_envs = [CageCls(**eval_cage_kwargs) for _ in range(n_eval_envs)]

    if step_transforms is not None:
        step_transform = Compose(step_transforms)

    eval_sampler = EvalSampler(
        max_traj_length=max_traj_length,
        min_trajectories=min_trajectories,
        envs=eval_envs,
        agent=agent,
        step_buffer=step_buffer,
        obs_transform=step_transform,
    )

    return eval_sampler, step_buffer
