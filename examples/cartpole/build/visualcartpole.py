import platform
from typing import Tuple

from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from hera_gym.wrappers import (
    PixelObservationWrapper,
    ResizeObservationWrapper,
    TorchifyImageWrapper,
)
from gym.wrappers import (
    TimeLimit,
)
from hera_gym.pyglet import set_default_pyglet_options


def make_visualcartpole(
    image_shape: Tuple[int, int] = (100, 150),
    max_episode_steps: int = 250,
    observe_state: bool = False,
) -> Env:

    headless = platform.system() == "Linux"
    set_default_pyglet_options(headless=headless)

    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # replace state observations with pixel observations
    env = PixelObservationWrapper(
        env,
        pixel_key="pixels" if observe_state else None,
        state_key="state" if observe_state else None,
    )

    # downsample observations
    env = ResizeObservationWrapper(
        env,
        shape=image_shape,
        pixel_observation_key="pixels" if observe_state else None,
    )

    # make channel dimension the leading dimension (torch convention)
    env = TorchifyImageWrapper(
        env,
        pixel_observation_key="pixels" if observe_state else None,
    )

    return env
