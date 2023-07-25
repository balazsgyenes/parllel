import platform
from typing import Tuple

import gymnasium as gym
from gymnasium.wrappers import (
    TimeLimit,
    ClipAction,
    GrayScaleObservation,
)
from gymnasium.envs.classic_control import CartPoleEnv

from hera_gym.wrappers import (
    SubprocessWrapper,
    PixelObservationWrapper,
    ResizeObservationWrapper,
    TorchifyImageWrapper,
    FrameStackWrapper,
)
from hera_gym.pyglet import set_default_pyglet_options


def build_visual_cartpole(
    action_type: str = "discrete",
    reward_type: str = "dense",
    image_shape: Tuple[int, int] = (100, 150),
    max_episode_steps: int = 250,
    color: bool = True,
    num_frames: int = 1,
    observe_state: bool = False,
    subprocess: bool = False,
    headless: bool = False,
) -> gym.Env:

    # headless pyglet requires a GPU (so does not work with xvfb-run)
    # quick heuristic for GPU availability is to check if OS is Linux
    headless = headless and platform.system() == "Linux"
    set_default_pyglet_options(headless=headless)

    env = ExtendedCartPoleEnv(
        action_type=action_type,
        reward_type=reward_type,
    )

    # add time limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if subprocess:
        # wrap in a subprocess so that rendering works properly
        env = SubprocessWrapper(env)

    # replace state observations with pixel observations
    env = PixelObservationWrapper(
        env,
        pixel_key="pixels" if observe_state else None,
        state_key="state" if observe_state else None,
    )

    if action_type == "continuous":
        # clip continuous actions outside of action space limits
        env = ClipAction(env)

    # downsample observations
    env = ResizeObservationWrapper(
        env,
        shape=image_shape,
        pixel_observation_key="pixels" if observe_state else None,
    )

    if not color:
        raise NotImplementedError
        env = GrayScaleObservation(env, keep_dim=True)

    # make channel dimension the leading dimension (torch convention)
    env = TorchifyImageWrapper(
        env,
        pixel_observation_key="pixels" if observe_state else None,
    )

    if num_frames > 1:
        env = FrameStackWrapper(env, num_stack=num_frames)

    return env