from typing import Sequence

import numpy as np
import pytest

from parllel import Array, ArrayDict, dict_map
from parllel.cages import Cage, MultiAgentTrajInfo, SerialCage, TrajInfo
from parllel.samplers.recurrent import RecurrentSampler
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.samplers.tests.test_basic import (
    N_BATCHES,
    action_space,
    batch_spec,
    get_bootstrap,
    max_decorrelation_steps,
    multireward,
    observation_space,
)
from parllel.tree.utils import assert_dict_equal


@pytest.fixture(params=[(8, 12), (6, 2)])
def episode_lengths(request, batch_spec):
    start, step = request.param
    return list(
        range(
            start,
            start + step * batch_spec.B,
            step,
        )
    )


@pytest.fixture
def envs(action_space, observation_space, batch_spec, multireward, episode_lengths):
    cages = [
        SerialCage(
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
        )
        for length in episode_lengths
    ]

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
    return agent


@pytest.fixture
def sample_tree(
    action_space,
    observation_space,
    batch_spec,
    envs: Sequence[Cage],
    agent,
    get_bootstrap,
):
    # get example output from env
    envs[0].random_step_async()
    action, next_obs, obs, reward, terminated, truncated, info = envs[0].await_step()
    agent_info = agent.get_agent_info()

    sample_tree: ArrayDict[Array] = ArrayDict()

    # allocate sample tree based on examples
    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        obs,
        batch_shape=tuple(batch_spec),
        padding=1,
    )
    sample_tree["reward"] = dict_map(
        Array.from_numpy,
        reward,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
    )
    sample_tree["terminated"] = Array.from_numpy(
        terminated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
    )
    sample_tree["truncated"] = Array.from_numpy(
        truncated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
    )
    sample_tree["done"] = Array.from_numpy(
        terminated,
        batch_shape=tuple(batch_spec),
        feature_shape=(),
        dtype=bool,
        padding=1,
    )
    sample_tree["valid"] = sample_tree["done"].new_array(storage="local")
    sample_tree["env_info"] = dict_map(
        Array.from_numpy,
        info,
        batch_shape=tuple(batch_spec),
    )
    sample_tree["action"] = dict_map(
        Array.from_numpy,
        action,
        batch_shape=tuple(batch_spec),
    )
    sample_tree["agent_info"] = dict_map(
        Array.from_numpy,
        agent_info,
        batch_shape=tuple(batch_spec),
    )
    sample_tree["initial_rnn_state"] = Array(
        feature_shape=(),
        batch_shape=(batch_spec.B,),
        dtype=np.float32,
    )

    if get_bootstrap:
        sample_tree["bootstrap_value"] = Array(
            feature_shape=(),
            batch_shape=(batch_spec.B,),
            dtype=np.float32,
        )

    yield sample_tree

    sample_tree.close()


@pytest.fixture(params=[RecurrentSampler])
def samples(
    request,
    batch_spec,
    envs,
    agent,
    sample_tree,
    max_decorrelation_steps,
    get_bootstrap,
):
    SamplerClass = request.param

    sampler = SamplerClass(
        batch_spec=batch_spec,
        envs=envs,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=max_decorrelation_steps,
        get_bootstrap_value=get_bootstrap,
    )

    batches, all_completed_trajs = list(), list()
    elapsed_steps = 0
    for _ in range(N_BATCHES):
        batch, completed_trajs = sampler.collect_batch(elapsed_steps)
        batches.append(batch.to_ndarray().copy())
        all_completed_trajs.append(completed_trajs)
        elapsed_steps += batch_spec.size

    yield batches, all_completed_trajs

    sampler.close()


class TestRecurrentSampler:
    def test_batches(
        self,
        samples,
        envs,
        agent,
        batch_spec,
        max_decorrelation_steps,
        get_bootstrap,
    ):
        batches, _ = samples

        # verify that agent was reset before very first batch
        assert np.all(agent.resets[-1])
        assert np.all(agent.states[0] == 0.0)

        if not max_decorrelation_steps:
            # verify that environments were all reset before very first batch
            # (only if not decorrelated)
            for env in envs:
                assert np.all(env._env.resets[-1])

        for i, batch in enumerate(batches):
            # verify rnn_states
            assert_dict_equal(batch["initial_rnn_state"], agent.initial_rnn_states[i])
            # remove because it cannot be indexed in time
            batch.pop("initial_rnn_state")

            if get_bootstrap:
                # check bootstrap values
                assert_dict_equal(batch["bootstrap_value"], agent.values[i])
                # remove them because they cannot be indexed in time
                batch.pop("bootstrap_value")

            time_slice = slice(i * batch_spec.T, (i + 1) * batch_spec.T)

            valid = np.asarray(batch["valid"])

            # verify that the agent samples are those generated by the agent
            assert_dict_equal(
                agent.samples[time_slice].to_ndarray()[valid],
                batch.to_ndarray()[valid],
                f"batch{(i+1)}.agent",
            )

            # check agent resets at the end of batch if a done occurred
            done_envs = np.any(batch["done"], axis=0)
            assert np.array_equal(agent.resets[time_slice][batch_spec.T - 1], done_envs)
            # check that the agent was only reset at the end of the batch
            assert not np.any(agent.resets[time_slice][: batch_spec.T - 1])

            # check that agent state is 0 at beginning of next batch for those
            # environments that were done during this batch
            agent_state_next_batch = np.asarray(agent.states[(i + 1) * batch_spec.T])
            assert np.all(agent_state_next_batch[done_envs] == 0)

            for b, env in enumerate(envs):
                b_valid = valid[:, b]

                # verify that steps before and including done are valid
                # and steps after done after invalid
                t_done = np.argwhere(env._env.samples["terminated"][time_slice])
                after_first_done = t_done[0, 0] + 1 if len(t_done) else batch_spec.T
                assert np.all(b_valid[:after_first_done])
                assert not np.any(b_valid[after_first_done:])

                # verify that the env samples are those generated by the env
                assert_dict_equal(
                    env._env.samples[time_slice].to_ndarray()[b_valid],
                    batch[:, b].to_ndarray()[b_valid],
                    f"batch{(i+1)}_env",
                )

                # verify that the env saw the correct action at each step
                assert_dict_equal(
                    dict_map(
                        np.asarray, env._env.samples["env_info"]["action"][time_slice]
                    )[b_valid],
                    dict_map(np.asarray, batch["action"][:, b])[b_valid],
                    f"batch{(i+1)}_action",
                )

                # verify that the agent saw the correct observation at each step
                assert_dict_equal(
                    dict_map(
                        np.asarray,
                        agent.samples["agent_info"]["observation"][time_slice, b],
                    )[b_valid],
                    dict_map(np.asarray, batch["observation"][:, b])[b_valid],
                    f"batch{(i+1)}_observation",
                )

                # check that environment was reset only on done
                ref_resets = env._env.resets[time_slice].to_ndarray()[b_valid]
                test_dones = batch["done"][:, b][b_valid]
                assert np.array_equal(ref_resets, test_dones)
