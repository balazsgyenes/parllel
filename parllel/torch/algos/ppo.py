from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from parllel import Array, ArrayDict
from parllel.algorithm import Algorithm
import parllel.logger as logger
from parllel.replays import BatchedDataLoader
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.agents.pg import PgPrediction
from parllel.torch.utils import valid_mean, explained_variance


class PPO(Algorithm):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.

    Also refer to
    - https://github.com/boschresearch/trust-region-layers/blob/main/trust_region_projections/algorithms/pg/pg.py
    - https://github.com/DLR-RM/stable-baselines3/blob/e75e1de4c127747527befc131d143361eddddae3/stable_baselines3/ppo/ppo.py
    """

    def __init__(
        self,
        agent: CategoricalPgAgent | GaussianPgAgent,
        dataloader: BatchedDataLoader[Tensor],
        optimizer: Optimizer,
        learning_rate_scheduler: _LRScheduler | None,
        value_loss_coeff: float,
        entropy_loss_coeff: float,
        clip_grad_norm: float | None,
        epochs: int,
        ratio_clip: float,
        value_clipping_mode: str,
        value_clip: float | None = None,
        kl_divergence_limit: float = np.inf,
        **kwargs,  # ignore additional arguments
    ) -> None:
        """Saves input settings."""
        self.agent = agent
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.ratio_clip = ratio_clip
        self.value_clipping_mode = value_clipping_mode
        self.value_clip = value_clip
        self.kl_divergence_limit = kl_divergence_limit

        self.update_counter = 0
        self.early_stopping = False
        self.algo_log_info: defaultdict[str, int | list[float]] = defaultdict(list)

    def optimize_agent(
        self,
        elapsed_steps: int,
        samples: ArrayDict[Array],
    ) -> dict[str, int | list[float]]:
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        self.agent.train_mode(elapsed_steps)

        self.algo_log_info.clear()
        self.early_stopping = False

        for _ in range(self.epochs):
            for batch in self.dataloader.batches():
                self.optimizer.zero_grad()
                loss = self.loss(batch)
                if self.early_stopping:
                    break
                loss.backward()
                if self.clip_grad_norm is not None:
                    # TODO: compute and log grad_norm even if not clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(),
                        self.clip_grad_norm,
                    )
                    self.algo_log_info["grad_norm"].append(grad_norm.item())
                self.optimizer.step()
                self.update_counter += 1

            if self.early_stopping:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.algo_log_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        self.algo_log_info["n_updates"] = self.update_counter

        return self.algo_log_info

    def loss(self, batch: ArrayDict[Tensor]) -> torch.Tensor:
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_prediction: PgPrediction = self.agent.predict(
            batch["observation"],
            batch["agent_info"],
            batch["init_rnn_state"],
        )
        dist_info, value = agent_prediction.dist_info, agent_prediction.value
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(
            batch["action"],
            old_dist_info=batch["old_dist_info"],
            new_dist_info=dist_info,
        )
        surr_1 = ratio * batch["advantage"]
        clipped_ratio = torch.clamp(ratio, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
        surr_2 = clipped_ratio * batch["advantage"]
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = -valid_mean(surrogate, batch["valid"])

        if self.value_clipping_mode == "none":
            # No clipping
            value_error = 0.5 * (value - batch["return_"]) ** 2
        elif self.value_clipping_mode == "ratio":
            # Clipping the value per time step in respect to the ratio between old and new values
            value_ratio = value / batch["old_values"]
            clipped_values = torch.where(
                value_ratio > 1.0 + self.value_clip,
                batch["old_values"] * (1.0 + self.value_clip),
                value,
            )
            clipped_values = torch.where(
                value_ratio < 1.0 - self.value_clip,
                batch["old_values"] * (1.0 - self.value_clip),
                clipped_values,
            )
            clipped_value_error = 0.5 * (clipped_values - batch["return_"]) ** 2
            standard_value_error = 0.5 * (value - batch["return_"]) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        elif self.value_clipping_mode == "delta":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch["old_values"] - self.value_clip),
                batch["old_values"] + self.value_clip,
            )
            value_error = 0.5 * (clipped_values - batch["return_"]) ** 2
        elif self.value_clipping_mode == "delta_max":
            # Clipping the value per time step with its original (old) value in the boundaries of value_clip
            clipped_values = torch.min(
                torch.max(value, batch["old_values"] - self.value_clip),
                batch["old_values"] + self.value_clip,
            )
            clipped_value_error = 0.5 * (clipped_values - batch["return_"]) ** 2
            standard_value_error = 0.5 * (value - batch["return_"]) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        else:
            raise ValueError(
                f"Invalid value clipping mode '{self.value_clipping_mode}'"
            )

        value_loss = self.value_loss_coeff * valid_mean(value_error, batch["valid"])

        entropy = dist.mean_entropy(dist_info, batch["valid"])
        entropy_loss = -self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        # Compute a low-variance estimate of the KL divergence to use for
        # stopping further updates after a KL divergence limit is reached.
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl_div = torch.mean(ratio - 1 - torch.log(ratio))
            if approx_kl_div >= self.kl_divergence_limit:
                self.early_stopping = True
                logger.info(
                    f"Reached the maximum KL divergence limit of {self.kl_divergence_limit} at step {self.update_counter}, stopping further updates."
                )
                return loss

            perplexity = dist.mean_perplexity(dist_info, batch["valid"])

            self.algo_log_info["loss"].append(loss.item())
            self.algo_log_info["policy_gradient_loss"].append(pi_loss.item())
            self.algo_log_info["approx_kl"].append(approx_kl_div.item())
            clip_fraction = ((ratio - 1).abs() > self.ratio_clip).float().mean().item()
            self.algo_log_info["clip_fraction"].append(clip_fraction)
            if "log_std" in dist_info:
                self.algo_log_info["policy_log_std"].append(
                    dist_info["log_std"].mean().item()
                )
            self.algo_log_info["entropy_loss"].append(entropy_loss.item())
            self.algo_log_info["entropy"].append(entropy.item())
            self.algo_log_info["perplexity"].append(perplexity.item())
            self.algo_log_info["value_loss"].append(value_loss.item())
            explained_var = explained_variance(value, batch["return_"])
            self.algo_log_info["explained_variance"].append(explained_var)

        return loss


def build_dataloader_buffer(
    sample_buffer: ArrayDict[Array],
    recurrent: bool = False,
) -> ArrayDict[Array]:
    dataloader_buffer = ArrayDict(
        {
            "observation": sample_buffer["observation"],
            "agent_info": sample_buffer["agent_info"],
            "action": sample_buffer["action"],
            "return_": sample_buffer["return_"],
            "advantage": sample_buffer["advantage"],
            "valid": sample_buffer["valid"] if recurrent else None,
            "old_dist_info": sample_buffer["agent_info"]["dist_info"],
            "old_values": sample_buffer["agent_info"]["value"],
            "init_rnn_state": sample_buffer["initial_rnn_state"] if recurrent else None,
        }
    )
    return dataloader_buffer
