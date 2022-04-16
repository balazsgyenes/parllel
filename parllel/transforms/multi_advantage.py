import enum

import numpy as np

from parllel.arrays import Array
from parllel.buffers import EnvSamples, NamedTuple, NamedArrayTupleClass, Samples

from .advantage import compute_discount_return, compute_gae_advantage
from .transform import BatchTransform
from .numpy import broadcast_left_to_right


class ProblemType(enum.Enum):
    # single reward, single value estimate
    single_critic = 0
    # single reward, unique value estimates for each distribution
    independent_critics = 1
    # unique rewards for each distribution, unique value estimates
    markov_game = 3


class EstimateMultiAgentAdvantage(BatchTransform):
    def __init__(self, discount: float, gae_lambda: float) -> None:
        """Computes per-agent advantage based on a shared reward and agent-
        specific value estimates. This wrapper should be used any time the
        agent has a MultiDistribution, either because it is an EnsembleAgent,
        or because it outputs e.g. both discrete and continuous actions.
        
        Requires fields:
            - .env.reward
            - .env.done
            - .agent.agent_info.value
            - .agent.bootstrap_value
        
        Adds fields:
            - .env.advantage
            - .env.return_

        :param discount: discount (gamma) for discounting rewards over time
        :param gae_lambda: lambda parameter for GAE algorithm
        """
        self._discount = discount
        self._lambda = gae_lambda
        if gae_lambda == 1.0:
            # GAE reduces to empirical discounted return
            self.estimator = compute_discount_return
        else:
            self.estimator = compute_gae_advantage

    def dry_run(self, batch_samples: Samples, ArrayCls: Array) -> Samples:
        # get convenient local references
        env_samples: EnvSamples = batch_samples.env
        reward = env_samples.reward
        value = batch_samples.agent.agent_info.value

        # create new NamedArrayTuple for env samples with additional fields
        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("advantage", "return_")
        )

        # determine number of reward values and value estimates
        if isinstance(value, NamedTuple):
            n_agents = len(value)
            advantage_shape = reward.shape + (n_agents,)
            return_shape = reward.shape + (n_agents,)
            if isinstance(reward, NamedTuple):
                self.problem_type = ProblemType.independent_critics
            else:
                self.problem_type = ProblemType.markov_game
        else:
            advantage_shape = reward.shape + (1,)
            return_shape = reward.shape
            self.problem_type = ProblemType.single_critic

        # allocate new Array objects for advantage and return_
        advantage = ArrayCls(shape=advantage_shape, dtype=reward.dtype)
        return_ = ArrayCls(shape=return_shape, dtype=reward.dtype)

        # package everything back into batch_samples
        env_samples = EnvSamplesClass(
            **env_samples._asdict(), advantage=advantage, return_=return_,
        )
        batch_samples = batch_samples._replace(env = env_samples)

        return batch_samples

    def __call__(self, batch_samples: Samples) -> Samples:
        """Cases:
        scalar reward, dict value (independent actor-critic agents)
        dict reward, dict value (stochastic game)
        scalar reward, scalar value (central critic, expand advantage)
        """
        reward = batch_samples.env.reward
        done = np.asarray(batch_samples.env.done)
        value = batch_samples.agent.agent_info.value
        bootstrap_value = batch_samples.agent.bootstrap_value
        advantage = np.asarray(batch_samples.env.advantage)
        return_ = np.asarray(batch_samples.env.return_)

        if self.problem_type is ProblemType.single_critic:
            value = np.asarray(value)
            bootstrap_value = np.asarray(bootstrap_value)
            reward = np.asarray(reward)
        else:
            # definitely need to stack value and bootstrap value
            value = np.stack(value, axis=-1)
            bootstrap_value = np.stack(bootstrap_value, axis=-1)

            if self.problem_type is ProblemType.markov_game:
                # stack rewards for each agent in the same order as in value
                # and bootstrap value. the subagents might not be defined in
                # the same order in the agent as they are in the environment

                reward = np.stack(
                    (
                        getattr(reward, agent_key)
                        for agent_key in value._fields
                    ),
                    axis=-1,
                )
            else:
                reward = np.asarray(reward)

        # add T dimension to bootstrap_value so it can be broadcast with
        # advantage and other arrays
        bootstrap_value = np.expand_dims(bootstrap_value, axis=0)

        # if missing, add singleton trailing dimensions to arrays until they
        # all have the same dimensionality
        advantage, reward, done, value = broadcast_left_to_right(
            advantage, reward, done, value
        )
        
        self.estimator(reward, value, done, bootstrap_value, self._discount,
            self._lambda, advantage, return_)
        return batch_samples
