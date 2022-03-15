import multiprocessing as mp

from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.wrappers import TimeLimit

from parllel.samplers.collections import AgentSamples, EnvSamples, Samples
from parllel.arrays import Array, RotatingArray, ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.buffers import buffer_method
from parllel.buffers.utils import buffer_from_dict_example
from parllel.cages import Cage, ParallelProcessCage
from parllel.handlers import Handler
from parllel.samplers import MiniSampler
from parllel.samplers.tests.random_agent import RandomAgent
from parllel.types.traj_info import TrajInfo


batch_T = 20
batch_B = 4
recurrent = False
parallel = True


def make_env(
    max_episode_steps: int = 250,
) -> Env:
    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def test_single_batch():

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
            EnvClass=make_env,
            env_kwargs={},
            TrajInfoClass=TrajInfo,
            traj_info_kwargs = {},
            wait_before_reset=recurrent
        )
        for _ in range(batch_B)
    ]

    # get example output from env
    example_env = cages[0]
    example_env_output = example_env.get_example_output()
    obs, reward, done, info = example_env_output

    # instantiate model and agent
    agent = RandomAgent(example_env.spaces.action)
    handler = Handler(agent=agent)
    # get example output from agent
    example_inputs = (obs, None)
    action, agent_info = agent.initialize(example_inputs=example_inputs, n_states=batch_B)

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, (batch_T, batch_B), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, (batch_T, batch_B), ArrayCls, name="reward", force_float32=True)
    batch_done = buffer_from_dict_example(done, (batch_T, batch_B), ArrayCls, name="done")
    batch_info = buffer_from_dict_example(info, (batch_T, batch_B), ArrayCls, name="envinfo")
    batch_env_samples = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    batch_action = buffer_from_dict_example(action, (batch_T, batch_B), ArrayCls, name="action")
    batch_agent_info = buffer_from_dict_example(agent_info, (batch_T, batch_B), ArrayCls, name="agentinfo")
    batch_agent_samples = AgentSamples(batch_action, batch_agent_info)
    batch_samples = Samples(batch_agent_samples, batch_env_samples)

    if parallel:
        for cage in cages:
            cage.set_samples_buffer(batch_samples)

    # initialize sampler
    sampler = MiniSampler(batch_T=batch_T, batch_B=batch_B, envs=cages, agent=handler,
        batch_buffer=batch_samples, get_bootstrap_value=False)


    samples, completed_trajectories = sampler.collect_batch(elapsed_steps=0)

    print(samples)
    print(completed_trajectories)


    # cleanup
    cages, handler, batch_samples = sampler.close()
    agent.close()
    for cage in cages:
        cage.close()
    buffer_method(batch_samples, "close")
    buffer_method(batch_samples, "destroy")


if __name__ == "__main__":
    # mp.set_start_method("fork")
    test_single_batch()
