import pytest

from gym import spaces
import numpy as np

from parllel.arrays import (Array, RotatingArray, buffer_from_example,
    buffer_from_dict_example)
from parllel.buffers import (AgentSamples, EnvSamples, NamedArrayTuple,
    NamedArrayTupleClass, Samples, buffer_method)
from parllel.cages import Cage, MultiAgentTrajInfo, TrajInfo
from parllel.types import BatchSpec

from parllel.samplers.basic import BasicSampler
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_handler import DummyHandler


N_BATCHES = 3

@pytest.fixture(
    params=[(32, 4), (32, 1), (1, 4)],
    ids=["batch=32x4", "batch=32x1", "batch=1x4"],
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
    scope="module",
)
def multireward(request):
    return request.param

@pytest.fixture(
    params=[False, True],
    ids=["nobootstrap", "bootstrapvalue"],
    scope="module",
)
def get_bootstrap(request):
    return request.param

@pytest.fixture
def envs(action_space, observation_space, batch_spec, multireward):
    episode_lengths = list(range(5, 5 + batch_spec.B))
    cages = [Cage(
        EnvClass=DummyEnv,
        env_kwargs=dict(
            action_space=action_space,
            observation_space=observation_space,
            episode_length=length,
            batch_spec=batch_spec,
            n_batches=N_BATCHES,
            multireward=multireward,
        ),
        TrajInfoClass=MultiAgentTrajInfo if multireward else TrajInfo,
        traj_info_kwargs={},
        wait_before_reset=False,
    ) for length in episode_lengths]

    yield cages

    for cage in cages:
        cage.close()

@pytest.fixture
def agent(action_space, observation_space, batch_spec):
    agent = DummyAgent(
        action_space=action_space,
        observation_space=observation_space,
        batch_spec=batch_spec,
        n_batches=N_BATCHES,
        recurrent=False,
    )
    handler = DummyHandler(agent)
    return handler

@pytest.fixture
def batch_buffer(action_space, observation_space, batch_spec, envs, agent, get_bootstrap):
    # get example output from env
    obs, reward, done, info = envs[0].get_example_output()
    agent_info = agent.dry_run()

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
    batch_agent_info = buffer_from_example(agent_info,
        tuple(batch_spec), Array, name="agentinfo")

    if get_bootstrap:
        AgentSamplesWBootstrap = NamedArrayTupleClass(
            typename = AgentSamples._typename,
            fields = AgentSamples._fields + ("bootstrap_value",)
        )
        batch_bootstrap_value = buffer_from_example(np.array(0, np.float32),
            (batch_spec.B,), Array)

        batch_agent = AgentSamplesWBootstrap(batch_action, batch_agent_info,
            batch_bootstrap_value)
    else:
        batch_agent = AgentSamples(batch_action, batch_agent_info)

    batch_buffer = Samples(batch_agent, batch_env)
        
    for env in envs:
        env.set_samples_buffer(batch_action, batch_observation, batch_reward,
            batch_done, batch_info)

    yield batch_buffer

    buffer_method(batch_buffer, "close")
    buffer_method(batch_buffer, "destroy")

@pytest.fixture(params=[BasicSampler])
def samples(request, batch_spec, envs, agent, batch_buffer,
        max_decorrelation_steps, get_bootstrap):
    SamplerClass = request.param

    sampler = SamplerClass(
        batch_spec=batch_spec,
        envs = envs,
        agent = agent,
        batch_buffer = batch_buffer,
        max_steps_decorrelate=max_decorrelation_steps,
        get_bootstrap_value=get_bootstrap,
    )

    batches, all_completed_trajs = list(), list()
    elapsed_steps = 0
    for _ in range(N_BATCHES):
        batch, completed_trajs = sampler.collect_batch(elapsed_steps)
        batches.append(buffer_method(batch, "copy"))
        all_completed_trajs.append(completed_trajs)
        elapsed_steps += np.prod(tuple(batch_spec))

    yield batches, all_completed_trajs

    sampler.close()


def buffer_equal(ref, test, /, name=""):
    if isinstance(ref, NamedArrayTuple): # non-leaf node
        for ref_elem, field in zip(ref, ref._fields):
            test_elem = getattr(test, field)
            buffer_equal(ref_elem, test_elem, name + "." + field)
    else:
        assert np.array_equal(ref, test), name

class TestBasicSampler:
    def test_batches(self, samples, envs, agent, batch_spec, max_decorrelation_steps, get_bootstrap):
        batches, _ = samples

        # verify that agent was reset before very first batch
        assert np.all(agent.resets[agent.resets.first - 1])
        assert np.all(agent.states[0] == 0.)

        if not max_decorrelation_steps:
            # verify that environments were all reset before very first batch
            # (only if not decorrelated)
            for b, env in enumerate(envs):
                assert np.all(env._env.resets[env._env.resets.first - 1])

        for i, batch in enumerate(batches):

            time_slice = slice(i * batch_spec.T, (i + 1) * batch_spec.T)

            ref_agent_samples = agent.samples[time_slice]

            # verify that the agent samples are those generated by the agent
            # if bootstrap value exists in batch.agent, it is ignored because
            # only fields in ref_agent_samples are checked
            buffer_equal(ref_agent_samples, batch.agent, f"batch{(i+1)}.agent")
            
            for b, env in enumerate(envs):
                ref_env_samples = env._env.samples[time_slice]

                # verify that the env samples are those generated by the env
                buffer_equal(ref_env_samples, batch.env[:, b], f"batch{(i+1)}_env")

                # verify that the env saw the correct action at each step
                buffer_equal(ref_env_samples.env_info.action, batch.agent.action[:, b],
                    f"batch{(i+1)}_action")

                # verify that the agent saw the correct observation at each step
                buffer_equal(ref_agent_samples.agent_info.observation[:, b],
                             batch.env.observation[:, b],
                             f"batch{(i+1)}_observation")

                # check that agent and environment were reset only on done
                ref_resets = env._env.resets[time_slice]
                ref_agent_resets = agent.resets[time_slice, b]
                test_dones = batch.env.done[:, b]
                assert np.array_equal(ref_resets, test_dones)
                assert np.array_equal(ref_agent_resets, test_dones)

                # check that agent state is always 0 after reset
                previous_time = slice(i * batch_spec.T - 1, (i + 1) * batch_spec.T - 1)
                previous_reset = env._env.resets[previous_time]
                assert np.all(agent.states[time_slice, b][previous_reset] == 0)

                # check that agent state is otherwise not 0
                assert not np.any(agent.states[time_slice, b][~np.asarray(previous_reset)] == 0)

            if get_bootstrap:
                # check bootstrap values
                buffer_equal(batch.agent.bootstrap_value, agent.values[i])
