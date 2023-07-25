from typing import Optional
import copy

from gym.spaces import Box, Dict
from gym import ObservationWrapper

import numpy as np


class TorchifyImageWrapper(ObservationWrapper):

    def __init__(self, env, pixel_observation_key: Optional[str] = None):
        super().__init__(env)

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

        obs_shape = image_observation_space.shape
        assert len(obs_shape) == 3, "Observation must be an image with channels."
        assert (
            obs_shape[2] in (1, 3, 4)
        ), "Observation must be a grayscale, rgb, or rgbd image."
        
        torchified_image_space = Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8)

        if self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
            self.observation_space.spaces[self._pixel_key] = torchified_image_space
            self.observation = self._observation_dict
        else:
            self.observation_space = torchified_image_space
            self.observation = self._observation_image

    def _observation_dict(self, observation):
        image = observation[self._pixel_key]
        observation[self._pixel_key] = self._torchify_image(image)
        return observation

    def _observation_image(self, observation):
        observation = self._torchify_image(observation)
        return observation

    def _torchify_image(self, image):
        # move color channel first
        image = np.moveaxis(image, -1, 0)
        # ensures array is contiguous in memory, i.e. all strides positive
        # this is required to convert numpy array to Torch tensor
        image = np.ascontiguousarray(image)
        
        return image
