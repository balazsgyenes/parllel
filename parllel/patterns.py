from contextlib import contextmanager
from typing import Callable, Dict, Tuple

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray,
    RotatingSharedMemoryArray, buffer_from_example, buffer_from_dict_example)
from parllel.buffers import AgentSamples, EnvSamples, NamedArrayTupleClass
from parllel.cages import Cage, ProcessCage
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
    example_env_output = example_cage.get_example_output()
    obs, reward, done, info = example_env_output
    action = example_cage.spaces.action.sample()

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


def add_initial_rnn_state(batch_buffer, batch_init_rnn):
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


def add_bootstrap_value(batch_buffer):
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


def add_valid(batch_buffer):
    batch_buffer_env: EnvSamples = batch_buffer.env
    done = batch_buffer_env.done

    EnvSamplesClass = NamedArrayTupleClass(
        typename = batch_buffer_env._typename,
        fields = batch_buffer_env._fields + ("valid",)
    )

    # allocate new Array objects for advantage and return_
    batch_valid = buffer_from_example(done)

    batch_buffer_env = EnvSamplesClass(
        **batch_buffer_env._asdict(), valid=batch_valid,
    )
    batch_buffer = batch_buffer._replace(env=batch_buffer_env)

    return batch_buffer
