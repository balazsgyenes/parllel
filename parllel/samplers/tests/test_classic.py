from parllel.samplers.collections import AgentSamples, EnvSamples, Samples
from parllel.arrays import Array, RotatingArray, buffer_from_example
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.samplers.recurrent import ClassicSampler
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.types.traj_info import TrajInfo


def build_sampler(batch_T: int, batch_B: int, recurrent: bool):

    # create dependencies of runner: sampler, agent, and algorithm
    sampler = ClassicSampler(batch_T=batch_T, batch_B=batch_B, reset_only_after_batch=recurrent)
    agent = DummyAgent()

    # create an extra cage for example generation
    example_env = Cage(EnvClass=DummyEnv,
            env_kwargs={"episode_length": 5},
            TrajInfoClass=TrajInfo,
            traj_info_kwargs = {},
            wait_before_reset=recurrent)
    example_env.initialize()

    # get action space from example env
    obs_space, action_space = example_env.spaces

    # get example output from both env and agent
    prev_action = action_space.sample()
    example_env.step_async(prev_action)
    example_env_output = example_env.await_step()
    obs, reward, done, info = example_env_output

    example_inputs = (obs, prev_action, reward)
    # instantiate model and agent
    action, agent_info = agent.initialize(example_inputs=example_inputs, n_states=batch_B)

    # discard example env
    example_env.close()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_example(obs, (batch_T, batch_B), RotatingArray, padding=1)
    batch_reward = buffer_from_example(reward, (batch_T, batch_B), Array)
    batch_done = buffer_from_example(done, (batch_T, batch_B), Array)
    batch_info = buffer_from_example(info, (batch_T, batch_B), Array)
    batch_env_samples = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    batch_action = buffer_from_example(action, (batch_T, batch_B), Array)
    batch_agent_info = buffer_from_example(agent_info, (batch_T, batch_B), Array)
    batch_agent_samples = AgentSamples(batch_action, batch_agent_info)
    batch_samples = Samples(batch_agent_samples, batch_env_samples)

    # create cages to manage environments
    episode_lengths = [5 + i for i in range(batch_B)]
    cages = [Cage(
            EnvClass=DummyEnv,
            env_kwargs={"episode_length": episode_length},
            TrajInfoClass=TrajInfo,
            traj_info_kwargs = {},
            wait_before_reset=recurrent)
        for episode_length in episode_lengths]
    
    for cage in cages:
        cage.initialize()

    handler = Handler(agent=agent)

    # initialize sampler
    sampler.initialize(agent=handler, envs=cages, batch_buffer=batch_samples)

    return sampler


def test_single_batch():
    sampler = build_sampler(20, 4, recurrent=False)

    samples, completed_trajectories = sampler.collect_batch(elapsed_steps=0)

    print(samples)
    print(completed_trajectories)


if __name__ == "__main__":
    test_single_batch()