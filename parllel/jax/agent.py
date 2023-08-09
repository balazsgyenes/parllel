import functools
from typing import Any, Callable

import flax
import jax
import numpy as np


@functools.partial(jax.jit, static_argnums=0)
def policy_action(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    state: np.ndarray,
):
    """
    Forward pass of the network

    :param apply_fn: [TODO:description]
    :param params: [TODO:description]
    :param state: [TODO:description]
    """
    out = apply_fn({"params": params}, state)
    return out
