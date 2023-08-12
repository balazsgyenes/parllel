import functools
from typing import Any, Callable

import flax
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from parllel import Array, ArrayTree, dict_map, ArrayDict
from parllel.agents.agent import Agent


class JaxAgent(Agent):
    def __init__(self, model, example_obs: ArrayTree[Array], distribution=None) -> None:
        self.model = model
        self.apply_fn = self.model.apply_fn
        self.distribution = distribution
        example_obs = example_obs.to_ndarray()  # type: ignore
        example_obs = dict_map(jnp.asarray, example_obs)

    def step(self, observation, *, state: TrainState, env_indices):
        observation = observation.to_ndarray()
        observation = dict_map(jnp.asarray, observation)
        observation = jax.device_put(observation)
        model_outputs = policy_action(self.apply_fn, state.params, observation)
        dist_params = model_outputs["dist_params"]
        agent_info = ArrayDict({"dist_params": dist_params})


@functools.partial(jax.jit, static_argnums=0)
def policy_action(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    observation: np.ndarray,
):
    """
    Forward pass of the network

    :param apply_fn: [TODO:description]
    :param params: [TODO:description]
    :param state: [TODO:description]
    """
    model_outputs = apply_fn({"params": params}, observation)
    return out
