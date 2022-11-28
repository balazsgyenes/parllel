from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from parllel.algorithm import Algorithm
from parllel.arrays import Array
from parllel.buffers import Samples, NamedArrayTupleClass
import parllel.logger as logger
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.agents.pg import AgentPrediction
from parllel.torch.utils import (buffer_to_device, valid_mean,
    explained_variance)


SamplesForLoss = NamedArrayTupleClass("SamplesForLoss", [
    "observation", "agent_info", "action", "return_", "advantage", "valid",
    "init_rnn_state",
    ],
)


class A2C(Algorithm):
    """
    Advantage Actor Critic algorithm (synchronous).  Trains the agent by
    taking one gradient step on each iteration of samples.
    """
    def __init__(
        self,
        agent: TorchAgent,
        batch_buffer: SamplesForLoss[torch.Tensor],
        optimizer: Optimizer,
        learning_rate_scheduler: Optional[_LRScheduler],
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        clip_grad_norm: float,
    ) -> None:
        """Saves input settings."""
        self.agent = agent
        self.batch_buffer = batch_buffer
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_grad_norm = clip_grad_norm

        self.update_counter = 0
        self.algo_log_info = defaultdict(list)

    def optimize_agent(self,
        elapsed_steps: int,
        samples: Samples[Array],
    ) -> Dict[str, Union[int, List[float]]]:
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        self.algo_log_info.clear()
        batch = buffer_to_device(self.batch_buffer, device=self.agent.device)
        self.agent.train_mode(elapsed_steps)
        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(),
            self.clip_grad_norm,
        )
        self.algo_log_info["grad_norm"].append(grad_norm.item())
        self.optimizer.step()
        self.update_counter += 1
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.algo_log_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        self.algo_log_info["n_updates"] = self.update_counter

        return self.algo_log_info

    def loss(self, batch: SamplesForLoss) -> torch.Tensor:
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_prediction: AgentPrediction = self.agent.predict(
            batch.observation, batch.agent_info, batch.init_rnn_state)
        dist_info, value = agent_prediction.dist_info, agent_prediction.value
        dist = self.agent.distribution

        logli = dist.log_likelihood(batch.action, dist_info)
        pi_loss = - valid_mean(logli * batch.advantage, batch.valid)

        value_error = 0.5 * (value - batch.return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, batch.valid)

        entropy = dist.mean_entropy(dist_info, batch.valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, batch.valid)

        self.algo_log_info["loss"].append(loss.item())
        self.algo_log_info["policy_gradient_loss"].append(pi_loss.item())
        if hasattr(dist_info, "log_std"):
            self.algo_log_info["policy_log_std"].append(dist_info.log_std.mean().item())
        self.algo_log_info["entropy_loss"].append(entropy_loss.item())
        self.algo_log_info["entropy"].append(entropy.item())
        self.algo_log_info["perplexity"].append(perplexity.item())
        self.algo_log_info["value_loss"].append(value_loss.item())
        explained_var = explained_variance(value, batch.return_)
        self.algo_log_info["explained_variance"].append(explained_var.item())

        return loss


def add_default_a2c_config(config: Dict) -> Dict:
    defaults = dict(
        learning_rate_scheduler=None,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
    )
    config["algo"] = defaults | config.get("algo", {})

    config["learning_rate"] = config.get("learning_rate", 1e-3)

    return config


def build_batch_buffer(
    sample_buffer: Samples,
    recurrent: bool = False,
) -> SamplesForLoss:
    from parllel.buffers import buffer_asarray
    from parllel.torch.utils import torchify_buffer

    dataloader_buffer = SamplesForLoss(
        observation=sample_buffer.env.observation,
        agent_info=sample_buffer.agent.agent_info,
        action=sample_buffer.agent.action,
        return_=sample_buffer.env.return_,
        advantage=sample_buffer.env.advantage,
        valid=sample_buffer.env.valid if recurrent else None,
        init_rnn_state=sample_buffer.agent.initial_rnn_state if recurrent else None,
    )
    dataloader_buffer = buffer_asarray(dataloader_buffer)
    dataloader_buffer = torchify_buffer(dataloader_buffer)
    return dataloader_buffer
