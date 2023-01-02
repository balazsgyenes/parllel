import pytest

import numpy as np

from parllel.arrays import (Array, RotatingArray, buffer_from_example,
    buffer_from_dict_example)
from parllel.buffers import (AgentSamples, EnvSamples, NamedArrayTuple,
    NamedArrayTupleClass, Samples, buffer_method, buffer_asarray)
from parllel.cages import Cage, MultiAgentTrajInfo, TrajInfo

from parllel.buffers.tests.utils import buffer_equal
from parllel.samplers.recurrent import RecurrentSampler
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_handler import DummyHandler
from parllel.samplers.tests.test_basic import (N_BATCHES, batch_spec,
    observation_space, action_space, max_decorrelation_steps, multireward,
    get_bootstrap)


@pytest.fixture(params=[(8, 12), (6, 2)])
def episode_lengths(request, batch_spec):
    start, step = request.param
    return list(range(
        start,
        start + step * batch_spec.B,
        step,
    ))

@pytest.fixture
def envs(action_space, observation_space, batch_spec, multireward, episode_lengths):
    cages = [Cage(
        EnvClass=DummyEnv,
        env_kwargs=dict(
            action_space=action_space,
            observation_space=observation_space,
            episode_length=length,
            batch_spec=batch_spec,
            n_batches=N_BATCHES,
            multireward=multireward,
            reset_automatically=False,
        ),
        TrajInfoClass=MultiAgentTrajInfo if multireward else TrajInfo,
        reset_automatically=False,
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
        recurrent=True,
    )
    handler = DummyHandler(agent)
    return handler

@pytest.fixture
def batch_buffer(action_space, observation_space, batch_spec, envs, agent, get_bootstrap):
    # get example output from env
    envs[0].random_step_async()
    action, obs, reward, done, info = envs[0].await_step()
    agent_info = agent.get_agent_info()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec),
        RotatingArray, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec),
        Array, name="reward")
    batch_done = buffer_from_dict_example(done, tuple(batch_spec),
        Array, name="done")
    batch_info = buffer_from_dict_example(info, tuple(batch_spec),
        Array, name="envinfo")
    batch_valid = buffer_from_dict_example(done, tuple(batch_spec),
        RotatingArray, name="valid")

    EnvSamplesWValid = NamedArrayTupleClass(
        typename = EnvSamples._typename,
        fields = EnvSamples._fields + ("valid",),
    )

    batch_env = EnvSamplesWValid(batch_observation, batch_reward,
        batch_done, batch_info, batch_valid)
    batch_action = buffer_from_dict_example(action,
        tuple(batch_spec), Array, name="action")
    batch_agent_info = buffer_from_example(agent_info,
        tuple(batch_spec), Array, name="agentinfo")

    AgentSamplesWRnnState = NamedArrayTupleClass(
        typename = AgentSamples._typename,
        fields = AgentSamples._fields + ("initial_rnn_state",)
    )
    batch_init_rnn = buffer_from_example(np.array(0, np.float32),
            (batch_spec.B,), Array)

    if get_bootstrap:
        AgentSamplesWBootstrap = NamedArrayTupleClass(
            typename = AgentSamplesWRnnState._typename,
            fields = AgentSamplesWRnnState._fields + ("bootstrap_value",)
        )
        batch_bootstrap_value = buffer_from_example(np.array(0, np.float32),
            (batch_spec.B,), Array)

        batch_agent = AgentSamplesWBootstrap(batch_action, batch_agent_info,
            batch_init_rnn, batch_bootstrap_value)
    else:
        batch_agent = AgentSamplesWRnnState(batch_action, batch_agent_info, batch_init_rnn)

    batch_buffer = Samples(batch_agent, batch_env)
        
    for env in envs:
        env.set_samples_buffer(batch_action, batch_observation, batch_reward,
            batch_done, batch_info)

    yield batch_buffer

    buffer_method(batch_buffer, "close")
    buffer_method(batch_buffer, "destroy")

@pytest.fixture(params=[RecurrentSampler])
def samples(request, batch_spec, envs, agent, batch_buffer,
        max_decorrelation_steps, get_bootstrap):
    SamplerClass = request.param

    sampler = SamplerClass(
        batch_spec=batch_spec,
        envs=envs,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=max_decorrelation_steps,
        get_bootstrap_value=get_bootstrap,
    )

    batches, all_completed_trajs = list(), list()
    elapsed_steps = 0
    for _ in range(N_BATCHES):
        batch, completed_trajs = sampler.collect_batch(elapsed_steps)
        batch = buffer_asarray(batch)
        batches.append(buffer_method(batch, "copy"))
        all_completed_trajs.append(completed_trajs)
        elapsed_steps += np.prod(tuple(batch_spec))

    yield batches, all_completed_trajs

    sampler.close()


class TestRecurrentSampler:
    def test_batches(self, samples, envs, agent, batch_spec, max_decorrelation_steps, get_bootstrap):
        batches, _ = samples

        # verify that agent was reset before very first batch
        assert np.all(agent.resets[agent.resets.first - 1])
        assert np.all(agent.states[0] == 0.)

        if not max_decorrelation_steps:
            # verify that environments were all reset before very first batch
            # (only if not decorrelated)
            for env in envs:
                assert np.all(env._env.resets[env._env.resets.first - 1])

        for i, batch in enumerate(batches):

            time_slice = slice(i * batch_spec.T, (i + 1) * batch_spec.T)

            valid = batch.env.valid

            # verify that the agent samples are those generated by the agent
            # if bootstrap value exists in batch.agent, it is ignored because
            # only fields in ref_agent_samples are checked
            assert buffer_equal(agent.samples.action[time_slice][valid],
                                batch.agent.action[valid],
                                f"batch{(i+1)}.agent")
            assert buffer_equal(agent.samples.agent_info[time_slice][valid],
                                batch.agent.agent_info[valid],
                                f"batch{(i+1)}.agent")

            # check agent resets at the end of batch if a done occurred
            done_envs = np.any(batch.env.done, axis=0)
            assert np.array_equal(agent.resets[time_slice][-1], done_envs)
            assert not np.any(agent.resets[time_slice][:-1])

            # check that agent state is 0 at beginning of next batch for those
            # environments that were done during this batch
            assert np.all(agent.states[(i + 1) * batch_spec.T, done_envs] == 0)

            # verify init_rnn_state
            assert buffer_equal(batch.agent.initial_rnn_state,
                agent.init_rnn_states[i])

            if get_bootstrap:
                # check bootstrap values
                assert buffer_equal(batch.agent.bootstrap_value, agent.values[i])
            
            for b, env in enumerate(envs):
                b_valid = valid[:, b]

                # verify that steps before and including done are valid
                # and steps after done after invalid
                t_done = np.argwhere(env._env.samples.done[time_slice])
                after_first_done = t_done[0, 0] + 1 if len(t_done) else batch_spec.T
                assert np.all(b_valid[:after_first_done])
                assert not np.any(b_valid[after_first_done:])

                # verify that the env samples are those generated by the env
                assert buffer_equal(env._env.samples[time_slice][b_valid],
                                    batch.env[:, b][b_valid],
                                    f"batch{(i+1)}_env")

                # verify that the env saw the correct action at each step
                assert buffer_equal(
                    env._env.samples.env_info.action[time_slice][b_valid],
                    batch.agent.action[:, b][b_valid],
                    f"batch{(i+1)}_action"
                )

                # verify that the agent saw the correct observation at each step
                assert buffer_equal(
                    agent.samples.agent_info.observation[time_slice, b][b_valid],
                    batch.env.observation[:, b][b_valid],
                    f"batch{(i+1)}_observation"
                )

                # check that environment was reset only on done
                ref_resets = env._env.resets[time_slice][b_valid]
                test_dones = batch.env.done[:, b][b_valid]
                assert np.array_equal(ref_resets, test_dones)
