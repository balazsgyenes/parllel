from __future__ import annotations

import functools
from collections import defaultdict
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax.typing import ArrayLike

from parllel import ArrayDict
from parllel.algorithm import Algorithm
from parllel.jax import agent
from parllel.replays.batched_dataloader import BatchedDataLoader


# @jax.jit
# @functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
# def gae_advantages():
#     pass
class PPO(Algorithm):
    def __init__(
        self,
        state: TrainState,
        dataloader: BatchedDataLoader,
        optimizer: Callable,
        clip_grad_norm: float,
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        epochs: int,
        ratio_clip: float,
        value_clipping_mode,
        value_clip: float | None = None,
    ):
        self.state = state
        self.dataloader = dataloader
        self.optimizer = (
            optimizer  # optimizer function already includes gradient clipping
        )
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.ratio_clip = ratio_clip
        self.value_clipping_mode = value_clipping_mode
        self.value_clip = value_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.algo_log_info: defaultdict[str, Any] = defaultdict(list)

    def optimize_agent(self, elapsed_steps: int, samples):
        for _ in range(self.epochs):
            for batch in self.dataloader.batches():
                self.state, loss = train_step(
                    self.state,
                    batch,
                    clip_param=self.ratio_clip,
                    vf_coeff=self.value_loss_coeff,
                    entropy_coeff=self.entropy_loss_coeff,
                )
                self.algo_log_info["loss"].append(loss)


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    batch,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    loss = 0.0
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(
        state.params, state.apply_fn, batch, clip_param, vf_coeff, entropy_coeff
    )
    state = state.apply_gradients(grads=grads)

    return state, loss


def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    minibatch: tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = agent.policy_action(apply_fn, params, states)
    values = values[:, 0]  # Convert shapes: (batch, 1) -> (batch,)
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)
    entropy = jnp.sum(-probs * log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_param, ratios, 1.0 + clip_param
    )
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


def create_train_state(
    params,
    model: nn.Module,
    learning_rate: float,
):
    tx = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return state
