from typing import Optional, Union

import numpy as np

from parllel.buffers import Buffer, buffer_map
from parllel.handlers import Handler, AgentStep
from parllel.torch.utils import numpify_buffer, torchify_buffer


class TorchHandler(Handler):
    def dry_run(self, n_states: int, observation: Buffer, previous_action: Optional[Buffer] = None,
                ) -> AgentStep:
        # TODO: preallocate a torch tensor version of the samples buffer
        # at runtime, just index into torch samples buffer instead of calling
        # from_numpy repeatedly, which might be slow
        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))
        observation, previous_action = torchify_buffer((observation, previous_action))

        example = self._agent.dry_run(n_states, observation, previous_action)

        example = numpify_buffer(example)

        return example

    def step(self, observation: Buffer, previous_action: Optional[Buffer] = None,
             *, env_indices: Union[int, slice] = ..., out_action: Buffer = None,
             out_agent_info: Buffer = None,
             ) -> Optional[AgentStep]:

        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))
        observation, previous_action = torchify_buffer((observation, previous_action))

        agent_step: AgentStep = self._agent.step(observation, previous_action,
                                                 env_indices=env_indices)

        agent_step = numpify_buffer(agent_step)

        if any(out is None for out in (out_action, out_agent_info)):
            return agent_step
        else:
            action, agent_info = agent_step
            out_action[:] = action
            out_agent_info[:] = agent_info

    def value(self, observation: Buffer, previous_action: Optional[Buffer] = None,
              *, out_value: Buffer = None,
              ) -> Optional[Buffer]:
        observation, previous_action = buffer_map(np.asarray,(observation, previous_action))
        observation, previous_action = torchify_buffer((observation, previous_action))

        value: Buffer = self._agent.value(observation, previous_action)

        value = numpify_buffer(value)

        if out_value is None:
            return value
        else:
            out_value[:] = value
