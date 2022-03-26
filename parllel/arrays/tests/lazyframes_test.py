import functools
from itertools import count

import gym
from gym import spaces
from gym.wrappers import FrameStack
import numpy as np
from numpy.random import default_rng
import pytest

from parllel.arrays.array import Array
from parllel.arrays.lazyframes import LazyFramesArray
from parllel.arrays.managedmemory import ManagedMemoryArray, RotatingManagedMemoryArray
from parllel.arrays.rotating import RotatingArray
from parllel.arrays.sharedmemory import RotatingSharedMemoryArray, SharedMemoryArray


@pytest.fixture(params=[
    LazyFramesArray,
    ], scope="module")
def ArrayClass(request):
    return request.param

@pytest.fixture(scope="module")
def shape():
    return (16, 8, 3, 96, 96)

@pytest.fixture(scope="module")
def frame_ndims():
    return 3

@pytest.fixture(params=[3, 4], ids=["depth=3", "depth=4"], scope="module")
def stack_depth(request):
    return request.param

@pytest.fixture(params=[np.int32, np.float32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture
def rng():
    return default_rng()

@pytest.fixture
def done_array(shape, frame_ndims, stack_depth):
    leading_dims = shape[:-frame_ndims]
    dones = RotatingArray(shape=leading_dims, dtype=np.bool_, padding=stack_depth)
    yield dones
    dones.close()
    dones.destroy()

@pytest.fixture
def blank_array(ArrayClass, shape, frame_ndims, stack_depth, dtype, done_array):
    array = ArrayClass(
        shape=shape,
        stack_depth=stack_depth,
        frame_ndims=frame_ndims,
        dtype=dtype,
        done=done_array,
        reset_mode="repeat",
    )
    yield array
    array.close()
    array.destroy()

@pytest.fixture
def np_array(shape, frame_ndims, stack_depth, dtype):
    full_shape = shape[:-frame_ndims] + (stack_depth,) + shape[-frame_ndims:]
    array = np.zeros(shape=full_shape, dtype=dtype)
    return array

class DummyEnv(gym.Env):
    def __init__(self, frame_shape, dtype, rng, counter_init=0):
        self._counter = count(counter_init)
        self._frame_shape = frame_shape
        self._frame_size = np.prod(frame_shape)
        self._dtype = dtype
        self._rng = rng
        self.observation_space = spaces.Box(0, 255, shape=frame_shape)

    def step(self, action):
        done = self._rng.random() > 0.90
        return self._next_obs(), 0, done, {}

    def reset(self):
        return self._next_obs()
        
    def _next_obs(self):
        t = next(self._counter)
        obs = np.arange(t*self._frame_size, (t+1)*self._frame_size,
                        dtype=self._dtype,
                        ).reshape(self._frame_shape)
        return obs

@pytest.fixture
def dataset(shape, frame_ndims, stack_depth, dtype, rng,
         done_array, blank_array, np_array):
    batch_done = done_array
    batch_obs = blank_array
    batch_np_obs = np_array
    frame_shape = shape[-frame_ndims:]
    batch_T = shape[0]
    batch_B = shape[1]
    envs = list()
    for b in range(batch_B):
        env = DummyEnv(frame_shape, dtype, rng, b * 1000)
        env = FrameStack(env, stack_depth)
        reset_obs = env.reset()
        batch_obs[0, b] = reset_obs
        batch_np_obs[0, b] = np.asarray(reset_obs)
        envs.append(env)

    for t in range(0, batch_T-1):
        for b, env in enumerate(envs):
            obs, _, done, _ = env.step(None)
            if done:
                obs = env.reset()
            batch_obs[t+1, b] = obs
            batch_np_obs[t+1, b] = np.asarray(obs)
            batch_done[t, b] = done
    return batch_done, batch_obs, batch_np_obs


class TestLazyFramesArray:
    def test_reconstruction(self, dataset):
        batch_done, batch_obs, batch_np_obs = dataset

        batch_obs = np.asarray(batch_obs)

        assert np.array_equal(batch_obs, batch_np_obs)
