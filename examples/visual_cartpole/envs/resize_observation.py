import copy
from typing import Optional

import numpy as np

from gym.spaces import Box, Dict
from gym import ObservationWrapper


class ResizeObservationWrapper(ObservationWrapper):
    """Downsample the image observation."""
    def __new__(cls, env, shape, pixel_observation_key: Optional[str] = None):
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        shape = tuple(shape)

        # if env already has requested observation shape, do not wrap it
        wrapped_observation_space = env.observation_space
        if pixel_observation_key is not None:
            assert isinstance(wrapped_observation_space, Dict)
            image_observation_space = wrapped_observation_space.spaces[pixel_observation_key]
        else:
            assert isinstance(wrapped_observation_space, Box)
            image_observation_space = wrapped_observation_space
        wrapped_obs_shape = image_observation_space.shape[:2]

        if shape == wrapped_obs_shape:
            return env
        else:
            return super().__new__(cls)
    
    def __init__(self, env, shape, pixel_observation_key: Optional[str] = None):
        super().__init__(env)
        global cv2
        import cv2
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        shape = tuple(shape)
        self.shape = shape

        wrapped_observation_space = env.observation_space
        if pixel_observation_key is not None:
            assert isinstance(wrapped_observation_space, Dict)
            self._observation_is_dict = True
            self._pixel_key = pixel_observation_key
            image_observation_space = wrapped_observation_space.spaces[self._pixel_key]
        else:
            assert isinstance(wrapped_observation_space, Box)
            self._observation_is_dict = False
            image_observation_space = wrapped_observation_space

        resized_image_space = Box(
            low=0,
            high=255,
            shape=self.shape + image_observation_space.shape[2:],
            dtype=np.uint8)

        if self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
            self.observation_space.spaces[self._pixel_key] = resized_image_space
            self.observation = self._observation_dict
        else:
            self.observation_space = resized_image_space
            self.observation = self._observation_image

    def _observation_image(self, observation):
        return self._resize_image(observation)

    def _observation_dict(self, observation):
        image = observation[self._pixel_key]
        observation[self._pixel_key] = self._resize_image(image)
        return observation

    def _resize_image(self, image):
        # cv2.resize takes dsize argument as (w,h)
        image = cv2.resize(image, dsize=self.shape[::-1], interpolation=cv2.INTER_NEAREST)
        return image
