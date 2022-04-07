import pytest

from gym import spaces
import numpy as np

from parllel.arrays import Array, RotatingArray, buffer_from_dict_example
from parllel.buffers import NamedArrayTuple, buffer_asarray, buffer_method
from parllel.cages import Cage, MultiAgentTrajInfo, TrajInfo
from parllel.buffers import Samples, AgentSamples, EnvSamples
from parllel.types import BatchSpec

from parllel.samplers.basic import BasicSampler
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_handler import DummyHandler


@pytest.fixture(
    params=[(64, 8), (64, 1), (1, 8)],
    ids=["batch=64x8", "batch=64x1", "batch=1x8"],
    scope="module",
)
def batch_spec(request):
    T, B = request.param
    return BatchSpec(T, B)

@pytest.fixture(
    params=[
        spaces.Box(-10, 10, (4,)),
        spaces.Dict({"alice": spaces.Box(-10, 10, (4,)), "bob": spaces.Box(-10, 10, (2,))}),
    ],
    ids=["obs=Box", "obs=Dict"],
    scope="module",
)
def observation_space(request):
    return request.param

@pytest.fixture(
    params=[
        spaces.Discrete(2),
        spaces.Box(-10, 10, (4,)),
        spaces.Dict({"alice": spaces.Box(-10, 10, (3,)), "bob": spaces.Box(-10, 10, (6,))}),
    ],
    ids=["action=Discrete", "action=Box", "action=Dict"],
    scope="module",
)
def action_space(request):
    return request.param

@pytest.fixture(
    params=[0, 25],
    ids=["decorrelation=0", "decorrelation=25"],
    scope="module",
)
def max_decorrelation_steps(request):
    return request.param

@pytest.fixture(
    params=[False, True],
    ids=["singleagent", "multiagent"],
)
def envs(request, action_space, observation_space, batch_spec):
    multireward = request.param
    episode_lengths = list(range(5, 5 + batch_spec.B))
    cages = [Cage(
        EnvClass=DummyEnv,
        env_kwargs=dict(
            action_space=action_space,
            observation_space=observation_space,
            episode_length=length,
            batch_spec=batch_spec,
            n_batches=2,
            multireward=multireward,
        ),
        TrajInfoClass=MultiAgentTrajInfo if multireward else TrajInfo ,
        traj_info_kwargs={},
        wait_before_reset=False,
    ) for length in episode_lengths]

    yield cages

    for cage in cages:
        cage.close()

@pytest.fixture()
def agent(action_space, observation_space, batch_spec):
    agent = DummyAgent(
        action_space=action_space,
        observation_space=observation_space,
        batch_spec=batch_spec,
        n_batches=2,
        recurrent=False,
    )
    handler = DummyHandler(agent)
    return handler

@pytest.fixture()
def batch_buffer(action_space, observation_space, batch_spec, envs):
    # get example output from env
    obs, reward, done, info = envs[0].get_example_output()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec),
        RotatingArray, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec),
        Array, name="reward")
    batch_done = buffer_from_dict_example(done, tuple(batch_spec),
        Array, name="done")
    batch_info = buffer_from_dict_example(info, tuple(batch_spec),
        Array, name="envinfo")
    batch_env = EnvSamples(batch_observation, batch_reward,
        batch_done, batch_info)
    batch_action = buffer_from_dict_example(action_space.sample(),
        tuple(batch_spec), Array, name="action")
    batch_agent_info = buffer_from_dict_example(
        {"observation": observation_space.sample()},
        tuple(batch_spec), Array, name="agentinfo")
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    for env in envs:
        env.set_samples_buffer(batch_action, batch_observation, batch_reward,
            batch_done, batch_info)

    yield batch_buffer

    buffer_method(batch_buffer, "close")
    buffer_method(batch_buffer, "destroy")

@pytest.fixture(params=[BasicSampler])
def samples(request, batch_spec, envs, agent, batch_buffer, max_decorrelation_steps):
    SamplerClass = request.param

    sampler = SamplerClass(
        batch_spec=batch_spec,
        envs = envs,
        agent = agent,
        batch_buffer = batch_buffer,
        max_steps_decorrelate=max_decorrelation_steps,
    )

    batch1, completed_trajs1 = sampler.collect_batch(0)
    batch1 = buffer_method(batch1, "copy")
    batch2, completed_trajs2 = sampler.collect_batch(np.prod(tuple(batch_spec)))
    batch2 = buffer_method(batch2, "copy")
    
    yield (batch1, batch2, completed_trajs1, completed_trajs2)

    sampler.close()


def buffer_equal(x, y, /, name=""):
    if isinstance(x, NamedArrayTuple): # non-leaf node
        return [buffer_equal(elem_x, elem_y, name + "." + field)
                for (elem_x, elem_y, field)
                in zip(x,y, x._fields)]
    
    assert np.array_equal(x, y), name

def check_batch(test_samples, ref_env_samples, ref_agent_samples, name):

    # verify that the samples the algorithm would see are the same
    buffer_equal(ref_env_samples, test_samples.env, name)

    # verify that the env saw the correct actions at each step
    buffer_equal(ref_env_samples.env_info.action, test_samples.agent.action,
        name + "_action")

    # verify that the agent saw the correct observation at each step
    buffer_equal(ref_agent_samples.agent_info.observation, test_samples.env.observation,
        name + "_action")

class TestBasicSampler:
    def test_batches(self, samples, envs, agent, batch_spec):
        batch1, batch2, _, _ = samples

        for b, env in enumerate(envs):
            # check first batch
            test_samples = batch1[:, b]
            ref_env_samples = buffer_asarray(env._env.samples[:batch_spec.T])
            ref_agent_samples = buffer_asarray(agent._samples[:batch_spec.T, b])
            check_batch(test_samples, ref_env_samples, ref_agent_samples, "batch1")

            # check second batch
            test_samples = batch2[:, b]
            ref_env_samples = buffer_asarray(env._env.samples[batch_spec.T:])
            ref_agent_samples = buffer_asarray(agent._samples[batch_spec.T:, b])
            check_batch(test_samples, ref_env_samples, ref_agent_samples, "batch2")

            # check resets to agent and environment

            # check bootstrap values

            # check that transforms see correct values

            # check trajectories
