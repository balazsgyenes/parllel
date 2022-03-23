from contextlib import contextmanager
from typing import Callable, Dict, Tuple

from parllel.buffers import buffer_method
from parllel.arrays import (Array, RotatingArray, SharedMemoryArray,
    RotatingSharedMemoryArray, buffer_from_example, buffer_from_dict_example)
from parllel.cages import Cage, ProcessCage
from parllel.samplers.collections import EnvSamples
from parllel.types import BatchSpec


@contextmanager
def build_cages_and_core_batch_buffers(
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
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec), ArrayCls, name="reward", force_float32=True)
    batch_done = buffer_from_dict_example(done, tuple(batch_spec), RotatingArrayCls, name="done", padding=1)
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), ArrayCls, name="envinfo")
    batch_env_samples = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)
    batch_action = buffer_from_example(action, tuple(batch_spec), ArrayCls)

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = (batch_action, batch_observation, batch_reward, batch_done, batch_info)
    
    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    try:
        yield cages, batch_action, batch_env_samples
    finally:
        for cage in cages:
            cage.close()
        buffer_method(batch_action, "close")
        buffer_method(batch_env_samples, "close")
        buffer_method(batch_action, "destroy")
        buffer_method(batch_env_samples, "destroy")
