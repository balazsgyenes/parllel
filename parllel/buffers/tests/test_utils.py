import pytest

import numpy as np

from parllel.buffers import NamedTupleClass, NamedArrayTupleClass
from parllel.buffers.tests.utils import buffer_equal
from parllel.buffers.utils import collate_buffers


class TestCollateBuffers:
    def test_collate_buffers(self):
        AgentStep = NamedTupleClass("AgentStep", ["action", "agent_info"])
        AgentInfo = NamedTupleClass("PgAgentInfo", [
            "dist_info",
            "value",
            "prev_action",
        ])

        subagent_steps = {}
        subagent_steps["agent1"] = AgentStep(
            action = np.array([1,2]),
            agent_info = AgentInfo(
                dist_info = np.array([[1, 2], [3, 4]]),
                value = np.array([7, 8, 9]),
                prev_action=None,
            )
        )
        subagent_steps["agent2"] = AgentStep(
            action = np.array([6]),
            agent_info = AgentInfo(
                dist_info = np.array([7, 8]),
                value = np.array([7, 8, 9]),
                prev_action=None,
            )
        )

        agent_step = collate_buffers(subagent_steps.values(),
            subagent_steps.keys())

        Agents = NamedTupleClass("Agents", ["agent1", "agent2"])

        ref_agent_step = AgentStep(
            action = Agents(
                agent1 = np.array([1,2]), agent2 = np.array([6]),
            ),
            agent_info = AgentInfo(
                dist_info = Agents(
                    agent1 = np.array([[1, 2], [3, 4]]),
                    agent2 = np.array([7, 8]),
                ),
                value = Agents(
                    agent1 = np.array([7, 8, 9]),
                    agent2 = np.array([7, 8, 9]),
                ),
                prev_action = Agents(
                    agent1 = None, agent2 = None,
                ),
            )
        )

        assert buffer_equal(ref_agent_step, agent_step)
