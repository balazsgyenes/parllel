from contextlib import contextmanager
import multiprocessing as mp
import time

import numpy as np

from parllel.buffers import buffer_from_example, buffer_from_dict_example, buffer_method
from parllel.arrays import Array, RotatingArray, ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.cages import Cage, ParallelProcessCage
from parllel.samplers.collections import Samples, AgentSamples, EnvSamples
from parllel.types import TrajInfo

from build.make_env import make_env


batch_B = 16
batch_T = 256
parallel = True
n_iterations = 50


@contextmanager
def build(EnvClass, env_kwargs=None, TrajInfoClass=TrajInfo, traj_info_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    if traj_info_kwargs is None:
        traj_info_kwargs = {}

    if parallel:
        CageCls = ParallelProcessCage
        ArrayCls = ManagedMemoryArray
        RotatingArrayCls = RotatingManagedMemoryArray
    else:
        CageCls = Cage
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    # create cages to manage environments
    cages = [
        CageCls(
            EnvClass = EnvClass,
            env_kwargs = env_kwargs,
            TrajInfoClass = TrajInfoClass,
            traj_info_kwargs = traj_info_kwargs,
            wait_before_reset = False, # reset immediately for speed test
        )
        for _ in range(batch_B)
    ]

    # get example output from env
    example_env_output = cages[0].get_example_output()
    obs, reward, done, info = example_env_output

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, (batch_T, batch_B), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, (batch_T, batch_B), ArrayCls, name="reward", force_float32=True)
    batch_done = buffer_from_dict_example(done, (batch_T, batch_B), ArrayCls, name="done")
    batch_info = buffer_from_dict_example(info, (batch_T, batch_B), ArrayCls, name="envinfo")
    batch_env_samples = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    # get example action from env
    example_action = cages[0].spaces.action.sample()

    # allocate batch buffer based on examples
    batch_action = buffer_from_example(example_action, (batch_T, batch_B), ArrayCls)
    batch_agent_samples = AgentSamples(batch_action, None)

    batch_samples = Samples(batch_agent_samples, batch_env_samples)

    if parallel:
        for cage in cages:
            cage.set_samples_buffer(batch_samples)

    try:
        yield cages, batch_samples
    
    finally:
        for cage in cages:
            cage.close()
        buffer_method(batch_samples, "close")
        buffer_method(batch_samples, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("spawn")

    env_kwargs = {
        "max_episode_steps": 1000,
    }

    with build(make_env, env_kwargs) as (envs, batch_samples):
        action, agent_info = (
            batch_samples.agent.action,
            batch_samples.agent.agent_info,
        )
        observation, reward, done, env_info = (
            batch_samples.env.observation,
            batch_samples.env.reward,
            batch_samples.env.done,
            batch_samples.env.env_info,
        )

        action_space = envs[0].spaces.action
        durations = np.zeros((batch_T,), dtype=float)

        for _ in range(n_iterations):
            for t in range(batch_T):
                # all environments receive the same actions
                action[t] = [action_space.sample()] * batch_B

                # don't include action space sampling in benchmark
                start = time.time()

                for b, env in enumerate(envs):
                    env.step_async(action[t, b],
                        out_obs=observation[t+1, b], out_reward=reward[t, b],
                        out_done=done[t, b], out_info=env_info[t, b])

                for b, env in enumerate(envs):
                    env.await_step()

                end = time.time()
                durations[t] = (end - start)

            print(f"Average step duration {durations.mean()*1000/batch_B:.4f} "
                    f"+/- {durations.std()*1000/batch_B:.4f} (ms) "
                    f"[{batch_B/durations.mean():.2f} FPS]")
