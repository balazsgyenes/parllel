import pytest

from gym import spaces
import numpy as np

from parllel.arrays import Array, RotatingArray, buffer_from_dict_example
from parllel.buffers import buffer_asarray, buffer_method
from parllel.cages import Cage
from parllel.samplers import Samples, AgentSamples, EnvSamples
from parllel.types import BatchSpec, TrajInfo

from parllel.samplers.basic import BasicSampler
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.samplers.tests.dummy_agent import DummyAgent, DummyHandler


@pytest.fixture(
    params=[(64, 8), (64, 1), (2, 8), (1, 8)],
    ids=["batch=64x8", "batch=64x1", "batch=2x8", "batch=1x8"],
    scope="module",
)
def batch_spec(request):
    T, B = request.param
    return BatchSpec(T, B)

@pytest.fixture(
    params=[spaces.Box(-np.inf, np.inf, (4,), np.float32)],
    ids=["obs=Box"],
    scope="module",
)
def observation_space(request):
    return request.param

@pytest.fixture(
    params=[spaces.Discrete(2)],
    ids=["action=Discrete"],
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
    params=[False],
    ids=["singleagent"],
)
def envs(request, action_space, observation_space, batch_spec):
    episode_lengths = list(range(5, 5 + batch_spec.B))
    cages = [Cage(
        EnvClass=DummyEnv,
        env_kwargs=dict(
            action_space=action_space,
            observation_space=observation_space,
            episode_length=length,
            batch_spec=batch_spec,
            n_batches=2,
            multi_reward=request.param,
        ),
        TrajInfoClass=TrajInfo,
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

@pytest.fixture(params=[BasicSampler]) # scope="class", 
def samples(request, batch_spec, envs, agent, batch_buffer, max_decorrelation_steps):
    SamplerClass = request.param

    sampler = SamplerClass(
        batch_spec=batch_spec,
        envs = envs,
        agent = agent,
        batch_buffer = batch_buffer,
        max_steps_decorrelate=max_decorrelation_steps,
    )

    samples1, completed_trajs1 = sampler.collect_batch(0)
    samples1 = buffer_method(samples1, "copy")
    samples2, completed_trajs2 = sampler.collect_batch(np.prod(tuple(batch_spec)))
    samples2 = buffer_method(samples2, "copy")
    
    yield (samples1, samples2, completed_trajs1, completed_trajs2)

    sampler.close()


class TestBasicSampler:
    def test_match(self, samples, envs, batch_spec):
        samples1, samples2, _, _ = samples

        for b, env in enumerate(envs):
            env_samples = samples1.env[:, b]
            reference = buffer_asarray(env._env.env_samples[:batch_spec.T])
            assert np.array_equal(reference.observation, env_samples.observation)
