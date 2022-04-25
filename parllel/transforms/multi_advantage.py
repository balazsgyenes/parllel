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
            - .agent.action
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
        action = batch_samples.agent.action
        value = np.asarray(batch_samples.agent.agent_info.value)

        if not isinstance(action, NamedTuple):
            raise TypeError("MultiAgent Advantage requires a dictionary action"
            " space.")

        # create new NamedArrayTuple for env samples with additional fields
        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("advantage", "return_")
        )

        # determine number of reward values and value estimates
        if value.ndim > 2:
            if isinstance(reward, NamedTuple):
                self.problem_type = ProblemType.markov_game
            else:
                self.problem_type = ProblemType.independent_critics
            advantage_shape = value.shape
        else:
            self.problem_type = ProblemType.single_critic
            # in algo, advantage must broadcast with distribution values (e.g.
            # log likelihood, likelihood ratio)
            advantage_shape = value.shape + (1,)

        # allocate new Array objects for advantage and return_
        advantage = ArrayCls(shape=advantage_shape, dtype=value.dtype)
        # in algo, return_ must broadcast with value
        return_ = ArrayCls(shape=value.shape, dtype=value.dtype)

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
        value = np.asarray(batch_samples.agent.agent_info.value)
        bootstrap_value = np.asarray(batch_samples.agent.bootstrap_value)
        advantage = np.asarray(batch_samples.env.advantage)
        return_ = np.asarray(batch_samples.env.return_)

        if self.problem_type is ProblemType.markov_game:
            # stack rewards for each agent in the same order as in value and
            # bootstrap value. the subagents might not be defined in the same
            # order in the agent as they are in the environment
            action = batch_samples.agent.action
            reward = np.stack(
                (getattr(reward, agent_key) for agent_key in action._fields),
                axis=-1,
            )
        else:
            reward = np.asarray(reward)

        # add T dimension to bootstrap_value so it can be broadcast with
        # advantage and other arrays
        bootstrap_value = np.expand_dims(bootstrap_value, axis=0)

        # if missing, add singleton trailing dimensions to arrays until they
        # all have the same dimensionality
        reward, value, done, bootstrap_value, advantage, return_ = broadcast_left_to_right(
            reward, value, done, bootstrap_value, advantage, return_
        )
        
        self.estimator(reward, value, done, bootstrap_value, self._discount,
            self._lambda, advantage, return_)
        return batch_samples
