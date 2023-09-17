import functools
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState

from parllel import Array, ArrayDict, ArrayTree, dict_map
from parllel.agents.agent import Agent

# class JaxAgent(Agent):
#     def __init__(self, model, example_obs: ArrayTree[Array], distribution=None) -> None:
#         self.model = model
#         self.apply_fn = self.model.apply_fn
#         self.distribution = distribution
#         example_obs = example_obs.to_ndarray()  # type: ignore
#         example_obs = dict_map(jnp.asarray, example_obs)

#     def step(self, observation, *, state: TrainState, env_indices):
#         observation = observation.to_ndarray()
#         observation = dict_map(jnp.asarray, observation)
#         observation = jax.device_put(observation)
#         model_outputs = policy_action(self.apply_fn, state.params, observation)
#         dist_params = model_outputs["dist_params"]
#         agent_info = ArrayDict({"dist_params": dist_params})


# @functools.partial(jax.jit, static_argnums=0)
def step(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    observation: Array,
    key: jax.random.PRNGKey,
):
    """
    Forward pass of the network

    :param apply_fn: [TODO:description]
    :param params: [TODO:description]
    :param state: [TODO:description]
    """
    observation = observation.to_ndarray()
    observation = dict_map(jnp.asarray, observation)
    model_outputs = apply_fn({"params": params}, observation)
    dist_params = model_outputs["dist_params"]
    logits = dist_params["probs"]
    # output action directly for now
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    u = jax.random.uniform(key, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log( -jnp.log(u)), axis=1)
    agent_info = ArrayDict({"dist_params": dist_params})
    agent_info["value"] = model_outputs["value"]
    return action, agent_info


def reset():
    pass

def reset_one(*args):
    pass
