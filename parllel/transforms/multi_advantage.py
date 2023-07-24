from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import numpy as np

from parllel import Array, ArrayDict
from parllel.np_utils import broadcast_left_to_right

from .advantage import compute_discount_return, compute_gae_advantage
from .transform import BatchTransform


class ProblemType(Enum):
    # single reward, single value estimate
    single_critic = 0
    # single reward, unique value estimates for each distribution
    independent_critics = 1
    # unique rewards for each distribution, unique value estimates
    markov_game = 3


class EstimateMultiAgentAdvantage(BatchTransform):
    """Computes per-agent advantage based on a shared reward and agent-specific
    value estimates. This wrapper should be used any time the agent has a
    MultiDistribution, either because it is an EnsembleAgent, or because it
    outputs e.g. both discrete and continuous actions.

    Requires input fields:
        - reward
        - done
        - action
        - agent_info["value"]
        - bootstrap_value

    Requires output fields:
        - advantage
        - return_

    :param sample_tree: the ArrayDict that will be passed to `__call__`.
    :param discount: discount (gamma) for discounting rewards over time
    :param gae_lambda: lambda parameter for GAE algorithm
    """

    def __init__(
        self,
        sample_tree: ArrayDict[Array],
        discount: float,
        gae_lambda: float,
    ) -> None:
        self._discount = discount
        self._lambda = gae_lambda
        if gae_lambda == 1.0:
            # GAE reduces to empirical discounted return
            self.estimator = compute_discount_return
        else:
            self.estimator = compute_gae_advantage

        # get convenient local references
        reward = sample_tree["reward"]
        value = sample_tree["agent_info"]["value"]
        advantage = sample_tree["advantage"]

        # determine number of reward values and value estimates
        if np.asarray(value).ndim > 2:
            if isinstance(reward, Mapping):
                self.problem_type = ProblemType.markov_game
            else:
                self.problem_type = ProblemType.independent_critics

            if advantage.shape != value.shape:
                raise ValueError("Advantage and Value must have the same shape")
        else:
            self.problem_type = ProblemType.single_critic
            if advantage.shape != (value.shape + (1,)):
                # in algo, advantage must broadcast with distribution values
                # (e.g. log likelihood, likelihood ratio)
                raise ValueError(
                    "Advantage must match shape of Value except for added trailing singleton dimension."
                )

    def __call__(self, sample_tree: ArrayDict[Array]) -> ArrayDict[Array]:
        """Cases:
        scalar reward, dict value (independent actor-critic agents)
        dict reward, dict value (markov game)
        scalar reward, scalar value (central critic, expand advantage)
        """
        reward = sample_tree["reward"]
        done = np.asarray(sample_tree["done"])
        value = np.asarray(sample_tree["agent_info"]["value"])
        bootstrap_value = np.asarray(sample_tree["bootstrap_value"])
        advantage = np.asarray(sample_tree["advantage"])
        return_ = np.asarray(sample_tree["return_"])

        if self.problem_type is ProblemType.markov_game:
            # stack rewards for each agent in the same order as in value and
            # bootstrap value. the subagents might not be defined in the same
            # order in the agent as they are in the environment
            action = sample_tree["action"]
            reward = np.stack(
                (reward[agent_key] for agent_key in action),
                axis=-1,
            )
        else:
            reward = np.asarray(reward)

        # add T dimension to bootstrap_value so it can be broadcast with
        # advantage and other arrays
        bootstrap_value = np.expand_dims(bootstrap_value, axis=0)

        # if missing, add singleton trailing dimensions to arrays until they
        # all have the same dimensionality
        (
            reward,
            value,
            done,
            bootstrap_value,
            advantage,
            return_,
        ) = broadcast_left_to_right(
            reward,
            value,
            done,
            bootstrap_value,
            advantage,
            return_,
        )

        self.estimator(
            reward,
            value,
            done,
            bootstrap_value,
            self._discount,
            self._lambda,
            advantage,
            return_,
        )
        return sample_tree
