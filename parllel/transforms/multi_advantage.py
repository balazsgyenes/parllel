import numpy as np

from parllel.arrays import Array
from parllel.buffers import EnvSamples, NamedArrayTupleClass, Samples

from .advantage import compute_discount_return, compute_gae_advantage
from .transform import BatchTransform
from .numpy import broadcast_across_trailing


class EstimatePerAgentAdvantage(BatchTransform):
    def __init__(self, discount: float, gae_lambda: float) -> None:
        """Computes per-agent advantage based on a shared reward and agent-
        specific value estimates.
        
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

        # create new NamedArrayTuple for env samples with additional fields
        EnvSamplesClass = NamedArrayTupleClass(
            typename = env_samples._typename,
            fields = env_samples._fields + ("advantage", "return_")
        )

        # allocate new Array objects for advantage and return_
        advantage = ArrayCls(shape=reward.shape, dtype=reward.dtype)
        return_ = ArrayCls(shape=reward.shape, dtype=reward.dtype)

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

        reward = (
            np.stack(reward, axis=-1)
            if isinstance(reward := np.asarray(batch_samples.env.reward), tuple)
            else np.asarray(reward)
        )
        done = np.asarray(batch_samples.env.done)
        value = (
            np.stack(value, axis=-1)
            if isinstance(value := batch_samples.agent.agent_info.value, tuple)
            else np.asarray(value)
        )
        reward, done, value = broadcast_across_trailing(reward, done, value)
        
        bootstrap_value = (
            np.stack(bootstrap_value, axis=-1)
            if isinstance(bootstrap_value := batch_samples.agent.bootstrap_value, tuple)
            else bootstrap_value
        )

        self.estimator(
            reward,
            value,
            done,
            bootstrap_value,
            self._discount,
            self._lambda,
            np.asarray(batch_samples.env.advantage),
            np.asarray(batch_samples.env.return_),
        )
        return batch_samples
