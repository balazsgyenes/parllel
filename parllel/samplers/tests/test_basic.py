from typing import Sequence

import numpy as np
import pytest
from gymnasium import spaces

from parllel import Array, ArrayDict, dict_map
from parllel.cages import Cage, MultiAgentTrajInfo, SerialCage, TrajInfo
from parllel.samplers.basic import BasicSampler
from parllel.samplers.tests.dummy_agent import DummyAgent
from parllel.samplers.tests.dummy_env import DummyEnv
from parllel.tree.utils import assert_dict_equal
from parllel.types import BatchSpec

N_BATCHES = 3
EPISODE_LENGTH_START = 8
EPISODE_LENGTH_STEP = 12


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
        spaces.Dict(
            {"alice": spaces.Box(-10, 10, (6,)), "bob": spaces.Box(-10, 10, (8,))}
        ),
    ],
    ids=["obs=Box", "obs=Dict"],
    scope="module",
)
def observation_space(request):
    return request.param


@pytest.fixture(
    params=[
        spaces.Discrete(2),
        spaces.Box(-10, 10, (3,)),
        spaces.Dict(
            {"alice": spaces.Box(-10, 10, (5,)), "bob": spaces.Box(-10, 10, (7,))}
        ),
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
    ids=["reward=single", "reward=multi"],
    scope="module",
)
def multireward(request):
    return request.param


@pytest.fixture(
    params=[False, True],
    ids=["bootstrap=False", "bootstrap=True"],
    scope="module",
)
def get_bootstrap(request):
    return request.param


@pytest.fixture
def envs(action_space, observation_space, batch_spec, multireward):
    episode_lengths = range(
        EPISODE_LENGTH_START,
        EPISODE_LENGTH_START + EPISODE_LENGTH_STEP * batch_spec.B,
        EPISODE_LENGTH_STEP,
    )
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
            ),
            TrajInfoClass=MultiAgentTrajInfo if multireward else TrajInfo,
            reset_automatically=True,
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
        recurrent=False,
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
    sample_tree["next_observation"] = dict_map(
        Array.from_numpy,
        next_obs,
        batch_shape=tuple(batch_spec),
    )
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

    if get_bootstrap:
        sample_tree["bootstrap_value"] = Array(
            feature_shape=(),
            batch_shape=(batch_spec.B,),
            dtype=np.float32,
        )

    yield sample_tree

    sample_tree.close()


@pytest.fixture(params=[BasicSampler])
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


class TestBasicSampler:
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
            if get_bootstrap:
                # check bootstrap values
                assert_dict_equal(batch["bootstrap_value"], agent.values[i])
                # remove them because they cannot be indexed in time
                batch.pop("bootstrap_value")

            time_slice = slice(i * batch_spec.T, (i + 1) * batch_spec.T)

            # verify that the agent samples are those generated by the agent
            assert_dict_equal(agent.samples[time_slice], batch, f"batch{(i+1)}_agent")

            # check that agent was reset only on done
            assert np.array_equal(agent.resets[time_slice], batch["done"])

            # check that agent state is always 0 after reset
            previous_time = slice(i * batch_spec.T - 1, (i + 1) * batch_spec.T - 1)
            previous_reset = agent.resets[previous_time]
            assert np.all(
                np.asarray(agent.states[time_slice])[np.asarray(previous_reset)] == 0
            )

            # check that agent state is otherwise not 0
            assert not np.any(
                np.asarray(agent.states[time_slice])[~np.asarray(previous_reset)] == 0
            )

            for b, env in enumerate(envs):
                # verify that the env samples are those generated by the env
                assert_dict_equal(
                    env._env.samples[time_slice], batch[:, b], f"batch{(i+1)}_env"
                )

                # verify that the env saw the correct action at each step
                assert_dict_equal(
                    env._env.samples["env_info"]["action"][time_slice],
                    batch["action"][:, b],
                    f"batch{(i+1)}_action",
                )

                # verify that the agent saw the correct observation at each step
                assert_dict_equal(
                    agent.samples["agent_info"]["observation"][time_slice, b],
                    batch["observation"][:, b],
                    f"batch{(i+1)}_observation",
                )

                # check that environment was reset only on done
                ref_resets = env._env.resets[time_slice]
                test_dones = batch["done"][:, b]
                assert np.array_equal(ref_resets, test_dones)
