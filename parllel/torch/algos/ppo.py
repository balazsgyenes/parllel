import torch
import torch.optim
import numpy as np

from parllel.buffers import NamedArrayTupleClass
from parllel.samplers import Samples
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.agents.pg import AgentPrediction
from parllel.torch.utils import buffer_to_device, torchify_buffer, valid_mean
from parllel.types import BatchSpec


PredictInputs = NamedArrayTupleClass("PredictInputs",
    ["observation", "agent_info"])


LossInputs = NamedArrayTupleClass("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info", "old_values"])


class PPO:
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
            batch_spec: BatchSpec,
            agent: TorchAgent,
            optimizer: torch.optim.Optimizer,
            learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
            value_loss_coeff: float = 1.,
            entropy_loss_coeff: float = 0.01,
            clip_grad_norm: float = 1.,
            minibatches: int = 4,
            epochs: int = 4,
            ratio_clip: float = 0.1,
            value_delta_clip: float = 0.1,
            value_ratio_clip: float = 0.1,
            value_clipping_mode: str = "none",
            ):
        """Saves input settings."""
        self.batch_spec = batch_spec
        self.agent = agent
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_grad_norm = clip_grad_norm
        self.minibatches = minibatches
        self.epochs = epochs
        self.ratio_clip = ratio_clip
        self.value_delta_clip = value_delta_clip
        self.value_ratio_clip = value_ratio_clip
        self.value_clipping_mode = value_clipping_mode

        self.rng = np.random.default_rng()

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def optimize_agent(self, elapsed_steps: int, samples: Samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """

        samples = torchify_buffer(samples)

        recurrent = self.agent.recurrent

        if recurrent:
            valid = samples.env.valid
            init_rnn_state = samples.agent.initial_rnn_state
        else:
            valid = None
            init_rnn_state = None

        # pack everything into NamedArrayTuples to enabling slicing
        agent_inputs = PredictInputs(
            observation=samples.env.observation,
            agent_info=samples.agent.agent_info,
        )
        loss_inputs = LossInputs(
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=samples.env.return_,
            advantage=samples.env.advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
            old_values=samples.agent.agent_info.value
        )
        # Move everything to device once, index there.
        # init_rnn_state is handled separately because it has no leading T dim
        loss_inputs, init_rnn_state = buffer_to_device(
            (loss_inputs, init_rnn_state), device=self.agent.device)

        T, B = self.batch_spec
        
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if recurrent else T * B
        minibatch_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in minibatch_indices(batch_size, minibatch_size, self.rng):
                if recurrent:
                    T_idxs = slice(None) # take entire trajectory
                    B_idxs = idxs # use shuffled indices for B dimension
                    minibatch_rnn_state = init_rnn_state[B_idxs]
                else:
                    B_idxs = idxs // T # index modulo T gives batch dimension
                    T_idxs = idxs % T # and remaining gives time step
                    minibatch_rnn_state = init_rnn_state
                    
                self.optimizer.zero_grad()
                # NOTE: if not recurrent, leading T and B dims are combined
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], minibatch_rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
             old_values, init_rnn_state):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_prediction: AgentPrediction = self.agent.predict(*agent_inputs, init_rnn_state)
        dist_info, value = agent_prediction.dist_info, agent_prediction.value
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        if self.value_clipping_mode == "ratio":
            # Clipping the value per time step in respect to the ratio between old and new values
            value_ratio = value / old_values
            clipped_values = torch.where(value_ratio > 1. + self.value_ratio_clip, old_values * (1. + self.value_ratio_clip), value)
            clipped_values = torch.where(value_ratio < 1. - self.value_ratio_clip, old_values * (1. - self.value_ratio_clip), clipped_values)
            clipped_value_error = 0.5 * (clipped_values - return_) ** 2
            standard_value_error = 0.5 * (value - return_) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        elif self.value_clipping_mode == "delta":
            # Clipping the value per time step with its original (old) value in the boundaries of value_delta_clip
            clipped_values = torch.min(torch.max(value, old_values - self.value_delta_clip), old_values + self.value_delta_clip)
            value_error = 0.5 * (clipped_values - return_) ** 2
        elif self.value_clipping_mode == "delta_max":
            # Clipping the value per time step with its original (old) value in the boundaries of value_delta_clip
            clipped_values = torch.min(torch.max(value, old_values - self.value_delta_clip), old_values + self.value_delta_clip)
            clipped_value_error = 0.5 * (clipped_values - return_) ** 2
            standard_value_error = 0.5 * (value - return_) ** 2
            value_error = torch.max(clipped_value_error, standard_value_error)
        else:
            # No clipping
            value_error = 0.5 * (value - return_) ** 2
        
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity


def minibatch_indices(data_length: int, minibatch_size: int, rng: np.random.Generator = None):
    """Yields minibatches of indexes, to use as a for-loop iterator, with
    option to shuffle.
    """
    if rng is not None:
        # shuffle 
        indexes = np.arange(data_length)
        rng.shuffle(indexes)
    # split the data into equally-sized pieces with size `minibatch_size`
    for start in range(0, data_length - minibatch_size + 1, minibatch_size):
        # create slice for the nth minibatch
        batch = slice(start, start + minibatch_size)
        if rng is not None:
            # if shuffling, 
            batch = indexes[batch]
        yield batch
