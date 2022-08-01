from contextlib import contextmanager
import datetime
import multiprocessing as mp

import numpy as np
import gym
from gym.envs.classic_control import CartPoleEnv
from gym import spaces
from gym.wrappers import TimeLimit

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray,
    RotatingSharedMemoryArray, buffer_from_dict_example)
from parllel.buffers import AgentSamples, buffer_method, EnvSamples, Samples
from parllel.cages import Cage, ProcessCage, TrajInfo
from parllel.cages.tests.dummy import DummyEnv
from parllel.cages.profiling import ProfilingProcessCage
from parllel.samplers.profiling import ProfilingSampler
from parllel.types import BatchSpec


def make_dummy_env(step_duration: float) -> gym.Env:
    observation_space = spaces.Box(0, 255, shape=[3, 192, 192], dtype=np.uint8)
    action_space = spaces.Discrete(n=2)
    env = DummyEnv(
        step_duration=step_duration,
        observation_space=observation_space,
        action_space=action_space,
    )
    # env = TimeLimit(env, max_episode_steps=100)
    return env


def make_cartpole_env(
    max_episode_steps: int = 250,
) -> gym.Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


@contextmanager
def build(config, parallel, profile_path):
    if parallel:
        if profile_path is not None:
            CageCls = ProfilingProcessCage
        else:
            CageCls = ProcessCage
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        CageCls = Cage
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    batch_spec: BatchSpec = config["sampler"]["batch_spec"]

    cage_kwargs = dict(
        EnvClass = config["env"]["EnvClass"],
        env_kwargs = config["env"]["env_kwargs"],
        TrajInfoClass = config["env"]["TrajInfoClass"],
        traj_info_kwargs = config["env"]["traj_info_kwargs"],
        wait_before_reset = False, # reset immediately for speed test
    )

    # create_example env
    example_cage = Cage(**cage_kwargs)

    # get example output from env
    example_cage.random_step_async()
    action, obs, reward, done, info = example_cage.await_step()
    
    example_cage.close()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec), ArrayCls, name="reward", force_32bit=True)
    batch_done = buffer_from_dict_example(done, tuple(batch_spec), RotatingArrayCls, name="done")
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), ArrayCls, name="envinfo")
    batch_buffer_env = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    batch_action = buffer_from_dict_example(action, tuple(batch_spec), ArrayCls,
        name="action", force_32bit=False)

    # pass batch buffers to Cage on creation
    if CageCls is ProcessCage:
        cage_kwargs["buffers"] = (batch_action, batch_observation, batch_reward, batch_done, batch_info)
    
    # create cages to manage environments
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]

    batch_agent_samples = AgentSamples(batch_action, None)

    batch_samples = Samples(batch_agent_samples, batch_buffer_env)

    sampler = ProfilingSampler(
        batch_spec = config["sampler"]["batch_spec"],
        envs = cages, 
        sample_buffer = batch_samples,
        n_iterations = config["sampler"]["n_iterations"],
        profile_path = profile_path,
    )

    try:
        yield sampler
    
    finally:
        for cage in cages:
            cage.close()
        buffer_method(batch_samples, "close")
        buffer_method(batch_samples, "destroy")
    

if __name__ == "__main__":
    import platform
    if platform.system() == "Darwin":
        mp.set_start_method("spawn")

    parallel = True
    profile_path = None
    # profile_path = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_main.profile"

    config = dict(
        env = dict(
            EnvClass = make_dummy_env,
            env_kwargs = {
                "step_duration": 0.03,
            },
            TrajInfoClass = TrajInfo,
            traj_info_kwargs = {},
        ),
        sampler = dict(
            batch_spec = BatchSpec(
                T = 256,
                B = 8,
            ),
            n_iterations = 1,
        )
    )

    for b in range(1, 16 + 1):

        config["sampler"]["batch_spec"] = BatchSpec(
            T = 64,
            B = b,
        )
        print(f"With {b} {'parallel' if parallel else 'serial'} environments:")
        
        with build(config, parallel, profile_path) as sampler:
            sampler.time_batches()
