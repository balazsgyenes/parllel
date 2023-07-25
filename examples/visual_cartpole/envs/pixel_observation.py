import collections
from typing import Optional, Tuple
from gym.spaces import Box, Dict
from gym import ObservationWrapper

import numpy as np

class PixelObservationWrapper(ObservationWrapper):
    """Extends or replaces observations with the output of an environment's
    `render` method.

    Args:
        env (gym.Env): environment to wrap.
        render_shape (Tuple[int]): expected shape of rendering returned by
            `render`. If None, `render` is called so the shape can be accessed.
            This can be a potentially destructive operation, since `reset` is
            also called, so the preferred method is to specify the shape
            explicitly.
        render_kwargs (dict): key-word arguments passed each time `render` is
            called on the environment.
        state_key (str): if None, the wrapped observation is discarded.
            Otherwise the wrapped observation is added to a dict with this key.
        pixel_key (str): if wrapped observations are retained in a dict, this
            cannot be None, and the output of `render` is added with this key.
    """
    def __init__(
        self,
        env,
        render_shape: Optional[Tuple[int]] = None,
        render_kwargs: dict = {"mode": "rgb_array"},
        state_key: Optional[str] = None,
        pixel_key: Optional[str] = "pixels",
    ):
        super().__init__(env)

        if render_shape is None:
            env.reset()
            image = env.render(**render_kwargs)
            render_shape = image.shape
        self._render_shape = render_shape
        self._render_kwargs = render_kwargs

        wrapped_observation_space = env.observation_space
        pixel_space = Box(
            low=0,
            high=255,
            shape=render_shape,
            dtype=np.uint8,
        )

        if state_key is not None:
            assert pixel_key is not None
            self.observation_space = Dict({
                pixel_key: pixel_space,
                state_key: wrapped_observation_space,
            })
        else:
            self.observation_space = pixel_space
        self._pixel_key = pixel_key
        self._state_key = state_key

    def observation(self, observation):
        image = self.env.render(**self._render_kwargs)
        assert image.shape == self._render_shape, (
            image.shape, self._render_shape)
        if self._state_key is not None:
            new_observation = collections.OrderedDict()
            new_observation[self._state_key] = observation
            new_observation[self._pixel_key] = image
            return new_observation
        else:
            return image
