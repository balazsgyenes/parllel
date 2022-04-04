import pytest

from parllel.arrays import Array, RotatingArray, buffer_from_dict_example
from parllel.cages import Cage
from parllel.samplers import Samples, EnvSamples
from parllel.types import BatchSpec, TrajInfo

from parllel.samplers.basic import BasicSampler
from parllel.samplers.tests.dummy_env import DummyEnv


@pytest.fixture(params=[
    (64, 8),
    (64, 1),
    (2, 8),
    (1, 8),
], scope="module")
def batch_spec(request):
    T, B = request.param
    return BatchSpec(T, B)

@pytest.fixture(scope="class")
def envs(batch_spec):
    return [Cage(
        EnvClass=DummyEnv,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        traj_info_kwargs={},
        wait_before_reset=False,
    ) for _ in range(batch_spec.B)]

@pytest.fixture(scope="class")
def batch_buffer(batch_spec, envs):
    ArrayCls = Array
    RotatingArrayCls = RotatingArray

    # create example env
    example_cage = envs[0]

    # get example output from env
    example_env_output = example_cage.get_example_output()
    obs, reward, done, info = example_env_output
    action = example_cage.spaces.action.sample()

    # allocate batch buffer based on examples
    batch_observation = buffer_from_dict_example(obs, tuple(batch_spec), RotatingArrayCls, name="obs", padding=1)
    batch_reward = buffer_from_dict_example(reward, tuple(batch_spec), ArrayCls, name="reward")
    batch_done = buffer_from_dict_example(done, tuple(batch_spec), RotatingArrayCls, name="done", padding=1)
    batch_info = buffer_from_dict_example(info, tuple(batch_spec), ArrayCls, name="envinfo")
    batch_buffer_env = EnvSamples(batch_observation, batch_reward, batch_done, batch_info)

    """In discrete problems, integer actions are used as array indices during
    optimization. Pytorch requires indices to be 64-bit integers, so we do not
    convert here.
    """
    batch_action = buffer_from_dict_example(action, tuple(batch_spec), ArrayCls, name="action", force_32bit=False)

    for env in envs:
        env.set_samples_buffer(batch_action, batch_observation, batch_reward,
            batch_done, batch_info)

    yield (batch_action, batch_buffer_env)

@pytest.fixture
def sampler(batch_spec, envs):
    BasicSampler(
        batch_spec=batch_spec,
        envs = envs
    )
