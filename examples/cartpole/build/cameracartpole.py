from typing import Tuple

from gym import Env
from gym.envs.classic_control.cartpole import CartPoleEnv
from hera_gym.wrappers import (
    PixelObservationWrapper,
    ResizeObservationWrapper,
    TorchifyImageWrapper,
    Virtual2DCameraWrapper,
)
from gym.wrappers import (
    TimeLimit,
)


def make_cameracartpole(
    camera_shape: Tuple[int, int] = (100, 150),
    image_shape: Tuple[int, int] = (100, 150),
    max_episode_steps: int = 250,
    observe_state: bool = False,
) -> Env:

    env = CartPoleEnv()

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # replace state observations with pixel observations
    env = PixelObservationWrapper(
        env,
        render_shape=(400, 600, 3),
        pixel_key="pixels" if observe_state else None,
        state_key="state" if observe_state else None,
    )

    # add virtual camera controlled by second agent
    env = Virtual2DCameraWrapper(
        env,
        camera_shape=camera_shape,
        pixel_observation_key="pixels" if observe_state else None,
        camera_action_key="camera",
        wrapped_action_key="cart",
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
