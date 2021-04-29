from typing import Optional

import numpy as np
from gym import Env, Wrapper
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.spaces import Dict as GymDict

from rlpyt.envs.base import EnvSpaces, EnvStep
from parllel.spaces.gym_wrapper import GymSpaceWrapper
from parllel.types.named_tuple import is_namedtuple, NamedTupleType, dict_to_namedtuple


class GymEnvWrapper(Wrapper):
    """Gym-style wrapper for converting the Openai Gym interface to the
    rlpyt interface.  Action and observation spaces are wrapped by rlpyt's
    ``GymSpaceWrapper``.

    Output `env_info` is automatically converted from a dictionary to a
    corresponding namedtuple, which the rlpyt sampler expects.  For this to
    work, every key that might appear in the gym environments `env_info` at
    any step must appear at the first step after a reset, as the `env_info`
    entries will have sampler memory pre-allocated for them (so they also
    cannot change dtype or shape). (see `EnvInfoWrapper`, `build_info_tuples`,
    and `info_to_nt` in file or more help/details)

    Warning:
        Unrecognized keys in `env_info` appearing later during use will be
        silently ignored.

    This wrapper looks for gym's ``TimeLimit`` env wrapper to
    see whether to add the field ``timeout`` to env info.   
    """

    def __init__(self,
        env: Env,
        act_null_value: float = 0,
        obs_null_value: float = 0,
        force_float32: bool = True,
    ):
        super().__init__(env)
        self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = GymSpaceWrapper(
            space=self.env.observation_space,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )

        # determine if environment also wrapped with gym's TimeLimit
        env_unwrapped = self.env
        time_limit = isinstance(self.env, GymTimeLimit)
        while not time_limit and hasattr(env_unwrapped, "env"):
            env_unwrapped = env_unwrapped.env
            time_limit = isinstance(env_unwrapped, GymTimeLimit)
        if time_limit:
            # gym's TimeLimit.truncated field has invalid name
            # replace with simple "timeout"
            info["timeout"] = info.pop("TimeLimit.truncated", False)
        self._time_limit = time_limit        

        # store all NamedTupleCls in a local variable
        self._namedtuple_classes = {}
        build_namedtuple_classes(info, "info", self._namedtuple_classes)

        if isinstance(r, dict):
            self._reward_is_dict = True
            build_namedtuple_classes(r, "reward", self._namedtuple_classes)
        else:
            self._reward_is_dict = False

    def step(self, action):
        """Reverts the action from rlpyt format to gym format (i.e. if composite-to-
        dictionary spaces), steps the gym environment, converts the observation
        from gym to rlpyt format (i.e. if dict-to-composite), and converts the
        env_info from dictionary into namedtuple."""
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        obs = self.observation_space.convert(o)
        if self._time_limit:
            info["timeout"] = info.pop("TimeLimit.truncated", False)
        info = dict_to_namedtuple(info, "info", self._namedtuple_classes)
        if self._reward_is_dict:
            r = dict_to_namedtuple(r, "reward", self._namedtuple_classes, force_ndarray=True)
        else:
            r = np.asanyarray(r, dtype=np.float32)
        return EnvStep(obs, r, d, info)

    def reset(self):
        """Returns converted observation from gym env reset."""
        return self.observation_space.convert(self.env.reset())

    @property
    def spaces(self):
        """Returns the rlpyt spaces for the wrapped env."""
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )


def build_namedtuple_classes(example: dict, name: str, classes: dict):
    NamedTupleCls = classes.get(name)
    # "." is not allowed if storing in globals
    keys = [str(k).replace(".", "_") for k in example.keys()]
    if NamedTupleCls is None:
        NamedTupleCls = NamedTupleType(name, keys)
        classes[name] = NamedTupleCls
    elif not (isinstance(NamedTupleCls, NamedTupleType) and
            sorted(NamedTupleCls._fields) == sorted(keys)):
        raise ValueError(f"Name clash in classes dict: {name}.")
    
    for k, v in example.items():
        if isinstance(v, dict):
            build_namedtuple_classes(v, "_".join([name, k]), classes)
