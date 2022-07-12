from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray,
    RotatingSharedMemoryArray, buffer_from_example, buffer_from_dict_example)
from parllel.buffers import (AgentSamples, EnvSamples, NamedArrayTupleClass,
    Samples, buffer_map)
from parllel.cages import Cage, ProcessCage
from parllel.handlers import Agent
from parllel.transforms import (Transform, ClipRewards, EstimateAdvantage,
    NormalizeAdvantage, NormalizeObservations, NormalizeRewards)
from parllel.types import BatchSpec


@contextmanager
def build_cages_and_env_buffers(
    EnvClass: Callable,
    env_kwargs: Dict,
    TrajInfoClass: Callable,
    traj_info_kwargs: Dict,
    wait_before_reset: bool,
    batch_spec: BatchSpec,
    parallel: bool,
) -> Tuple:

    if parallel:
        CageCls = ProcessCage
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        CageCls = Cage
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    cage_kwargs = dict(
        EnvClass = EnvClass,
        env_kwargs = env_kwargs,
        TrajInfoClass = TrajInfoClass,
        traj_info_kwargs = traj_info_kwargs,
        wait_before_reset = wait_before_reset,
    )

    # create example env
    example_cage = Cage(**cage_kwargs)

    # get example output from env
    example_cage.random_step_async()
    action, obs, reward, done, info = example_cage.await_step()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec), ArrayCls, name="reward")
    batch_done = buffer_from_dict_example(done, tuple(batch_spec), RotatingArrayCls, name="done", padding=1)
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), ArrayCls, name="envinfo")
    batch_buffer_env = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    """In discrete problems, integer actions are used as array indices during
    optimization. Pytorch requires indices to be 64-bit integers, so we do not
    convert here.
    """
    batch_action = buffer_from_dict_example(action, tuple(batch_spec), ArrayCls, name="action", force_32bit=False)

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = (batch_action, batch_observation, batch_reward, batch_done, batch_info)
    
    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    try:
        yield cages, batch_action, batch_buffer_env
    finally:
        for cage in cages:
            cage.close()


def add_initial_rnn_state(batch_buffer: Samples, agent: Agent):

    # get the Array type used for the rewards. reward might be a named tuple,
    # but the underlying array should be non-rotating
    # TODO: replace with some sane allocation rules
    types = buffer_map(type, batch_buffer.env.reward)
    while isinstance(types, tuple):
        types = types[0]
    ArrayCls = types

    rnn_state = agent.initial_rnn_state()
    batch_init_rnn = buffer_from_example(rnn_state, (), ArrayCls)
    
    batch_agent: AgentSamples = batch_buffer.agent    

    AgentSamplesClass = NamedArrayTupleClass(
        typename = batch_agent._typename,
        fields = batch_agent._fields + ("initial_rnn_state",)
    )

    batch_agent = AgentSamplesClass(
        **batch_agent._asdict(), initial_rnn_state=batch_init_rnn,
    )
    batch_buffer = batch_buffer._replace(agent=batch_agent)
    
    return batch_buffer


def add_bootstrap_value(batch_buffer: Samples):
    batch_agent: AgentSamples = batch_buffer.agent    
    batch_agent_info = batch_agent.agent_info

    AgentSamplesClass = NamedArrayTupleClass(
        typename = batch_agent._typename,
        fields = batch_agent._fields + ("bootstrap_value",)
    )

    # remove T dimension, creating an array with only (B,) leading dimensions
    batch_bootstrap_value = buffer_from_example(batch_agent_info.value[0])

    batch_agent = AgentSamplesClass(
        **batch_agent._asdict(), bootstrap_value=batch_bootstrap_value,
    )
    batch_buffer = batch_buffer._replace(agent=batch_agent)
    
    return batch_buffer


def add_valid(batch_buffer: Samples):
    batch_buffer_env: EnvSamples = batch_buffer.env
    done = batch_buffer_env.done

    EnvSamplesClass = NamedArrayTupleClass(
        typename = batch_buffer_env._typename,
        fields = batch_buffer_env._fields + ("valid",)
    )

    # allocate new Array objects for valid
    batch_valid = buffer_from_example(done)

    batch_buffer_env = EnvSamplesClass(
        **batch_buffer_env._asdict(), valid=batch_valid,
    )
    batch_buffer = batch_buffer._replace(env=batch_buffer_env)

    return batch_buffer


def add_advantage_estimation(
        batch_buffer: Samples,
        transforms: List[Transform],
        discount: float,
        gae_lambda: float,
        normalize: bool = False,
    ) -> Tuple[Samples, List[Transform]]:
    
    # add required fields to batch_buffer
    # get convenient local references
    env_samples: EnvSamples = batch_buffer.env
    reward = env_samples.reward

    # create new NamedArrayTuple for env samples with additional fields
    EnvSamplesClass = NamedArrayTupleClass(
        typename = env_samples._typename,
        fields = env_samples._fields + ("advantage", "return_")
    )

    # allocate new Array objects for advantage and return_
    batch_advantage = buffer_from_example(reward)
    batch_return_ = buffer_from_example(reward)

    # package everything back into batch_buffer
    env_samples = EnvSamplesClass(
        **env_samples._asdict(), advantage=batch_advantage, return_=batch_return_,
    )
    batch_buffer = batch_buffer._replace(env = env_samples)

    # create required transforms and add to list
    transforms.append(EstimateAdvantage(discount=discount,
        gae_lambda=gae_lambda))

    if normalize:
        transforms.append(NormalizeAdvantage(
            only_valid=hasattr(batch_buffer.env, "valid"),
        ))

    return batch_buffer, transforms


def add_obs_normalization(
        batch_buffer: Samples,
        transforms: List[Transform],
        initial_count: Union[int, float, None] = None,
    ) -> Tuple[Samples, List[Transform]]:

    transforms.append(
        NormalizeObservations(
            # get shape of observation assuming 2 leading dimensions
            obs_shape=batch_buffer.env.observation.shape[2:],
            only_valid=hasattr(batch_buffer.env, "valid"),
            initial_count=initial_count,
        )
    )

    return batch_buffer, transforms


def add_reward_normalization(
        batch_buffer: Samples,
        transforms: List[Transform],
        discount: float,
        initial_count: Union[int, float, None] = None,
    ) -> Tuple[Samples, List[Transform]]:

    # add "past_return_" field to batch_buffer
    # get convenient local references
    env_samples: EnvSamples = batch_buffer.env
    reward = env_samples.reward

    if not isinstance(env_samples.done, RotatingArray):
        raise TypeError("batch_buffer.env.done must be a RotatingArray "
                        "when using NormalizeRewards")

    # create new NamedArrayTuple for env samples with additional field
    EnvSamplesClass = NamedArrayTupleClass(
        typename = env_samples._typename,
        fields = env_samples._fields + ("past_return",)
    )

    # allocate new Array for past discounted returns
    # TODO: add smarter allocation rules here
    RotatingArrayCls = type(env_samples.done)
    batch_past_return = RotatingArrayCls(shape=reward.shape, dtype=reward.dtype)

    # package everything back into batch_buffer
    env_samples = EnvSamplesClass(
        **env_samples._asdict(), past_return=batch_past_return,
    )
    batch_buffer = batch_buffer._replace(env=env_samples)

    # create NormalizeReward transform and add to list
    transforms.append(
        NormalizeRewards(
            discount=discount,
            only_valid=hasattr(batch_buffer.env, "valid"),
            initial_count=initial_count,
        )
    )

    return batch_buffer, transforms


def add_reward_clipping(
        batch_buffer: Samples,
        transforms: List[Transform],
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> Tuple[Samples, List[Transform]]:

    transforms.append(
        ClipRewards(
            reward_min=reward_min,
            reward_max=reward_max,
        )
    )

    return batch_buffer, transforms
