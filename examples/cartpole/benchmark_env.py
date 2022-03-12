from contextlib import contextmanager
import datetime
import multiprocessing as mp

from parllel.buffers import buffer_from_example, buffer_from_dict_example, buffer_method
from parllel.arrays import Array, RotatingArray, ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.cages import Cage
from parllel.cages.profiler import ParallelProcessCage, ProfilingParallelProcessCage
from parllel.samplers.collections import Samples, AgentSamples, EnvSamples
from parllel.samplers.profiler import ProfilingSampler
from parllel.types import BatchSpec, TrajInfo

from build.make_env import make_env


batch_B = 16
batch_T = 256
batch_spec = BatchSpec(batch_T, batch_B)
parallel = True
n_iterations = 10
profile = False
profile_path = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_main.profile"


@contextmanager
def build(EnvClass, env_kwargs=None, TrajInfoClass=TrajInfo, traj_info_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    if traj_info_kwargs is None:
        traj_info_kwargs = {}

    if parallel:
        if profile:
            CageCls = ProfilingParallelProcessCage
        else:
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
        for _ in range(batch_spec.B)
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

    for cage in cages:
        cage.register_samples_buffer(batch_samples)

    sampler = ProfilingSampler(batch_spec, cages, batch_samples, n_iterations,
        profile_path=profile_path if profile else None)

    try:
        yield sampler
    
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

    with build(make_env, env_kwargs) as sampler:
        sampler.time_batches()