import torch
from torch import Tensor

from parllel import Array, ArrayDict, ArrayTree, Index, dict_map
from parllel.torch.agents.sac_agent import PiModelOutputs, SacAgent


class PointCloudSacAgent(SacAgent):
    def encode(self, observation: ArrayTree[Tensor]) -> ArrayTree[Tensor]:
        return self.model["encoder"](observation)

    @torch.no_grad()
    def step(
        self,
        observation: ArrayTree[Array],
        *,
        env_indices: Index = ...,
    ) -> tuple[Tensor, ArrayDict[Tensor]]:
        observation = observation.to_ndarray()
        observation = dict_map(torch.from_numpy, observation)
        observation = observation.to(device=self.device)
        encoding = self.model["encoder"](observation)
        dist_params: PiModelOutputs = self.model["pi"](encoding)
        action = self.distribution.sample(dist_params)
        return action.cpu(), ArrayDict()

    def q(
        self,
        encoding: ArrayTree[Tensor],
        action: ArrayTree[Tensor],
    ) -> tuple[Tensor, Tensor]:
        # We detach the encoding from its computational graph, since we only
        # want to optimize the encoder using the gradients from the policy
        # network.
        return super().q(encoding.detach(), action)
