from parllel.samplers.collections import AgentSamples, EnvSamples, Samples
from parllel.arrays import Array, RotatingArray
from parllel.buffers.utils import buffer_from_example
from parllel.cages import Cage
from parllel.handlers import Handler
from parllel.samplers import MiniSampler
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.types.traj_info import TrajInfo


def build_sampler(batch_T: int, batch_B: int, recurrent: bool):

    # create cages to manage environments
    episode_lengths = [5 + i for i in range(batch_B)]
    cages = [
        Cage(
            EnvClass=DummyEnv,
            env_kwargs={"episode_length": episode_length},
            TrajInfoClass=TrajInfo,
            traj_info_kwargs = {},
            wait_before_reset=recurrent
        )
        for episode_length in episode_lengths
    ]

    # get example output from env
    example_env = cages[0]
    example_env_output = example_env.get_example_output()
    obs, reward, done, info = example_env_output

    # instantiate model and agent
    agent = DummyAgent()
    handler = Handler(agent=agent)
    # get example output from agent
    example_inputs = (obs, None)
    action, agent_info = agent.initialize(example_inputs=example_inputs, n_states=batch_B)

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

    # initialize sampler
    sampler = MiniSampler(batch_T=batch_T, batch_B=batch_B, envs=cages, agent=handler,
        batch_buffer=batch_samples, get_bootstrap_value=False)

    return sampler


def test_single_batch():
    sampler = build_sampler(20, 4, recurrent=False)

    samples, completed_trajectories = sampler.collect_batch(elapsed_steps=0)

    print(samples)
    print(completed_trajectories)


if __name__ == "__main__":
    test_single_batch()