from parllel.handlers.agent import Agent
from parllel.samplers import EvalSampler

from .runner import Runner


class ShowPolicy(Runner):
    def __init__(self,
        sampler: EvalSampler,
        agent: Agent,
    ) -> None:

        super().__init__(
            log_dir=None,
        )

        self.sampler = sampler
        self.agent = agent

    def run(self) -> None:
        
        eval_trajs = self.sampler.collect_batch(elapsed_steps=0)

        # does not log, but at least prints to stdout
        self.log_progress(elapsed_steps=0, trajectories=eval_trajs)
