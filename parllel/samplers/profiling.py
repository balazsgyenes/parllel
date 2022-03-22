import cProfile
from pathlib import Path
import time
from typing import Optional, Sequence

import numpy as np

from parllel.cages import Cage
from parllel.types import BatchSpec

from .collections import Samples
from .mini import MiniSampler


class ProfilingSampler(MiniSampler):
    def __init__(self, batch_spec: BatchSpec, envs: Sequence[Cage], batch_buffer: Samples,
                n_iterations: int, profile_path: Optional[Path] = None
                ) -> None:
        super().__init__(batch_spec, envs, agent=None, batch_buffer=batch_buffer,
            get_bootstrap_value=False)

        self.n_iterations = n_iterations
        self.profile_path = profile_path

        self.action_space = envs[0].spaces.action
        self.durations = np.zeros((batch_spec.T,), dtype=float)
        self.profiler = cProfile.Profile() if profile_path is not None else None
    
    def reset_all(self) -> None:
        """Reset all environments and save to beginning of observation
        buffer.
        """
        observation = self.batch_buffer.env.observation
        for b, env in enumerate(self.envs):
            # save reset observation to the end of buffer, since it will be 
            # rotated to the beginning
            env.reset_async(out_obs=observation[observation.end + 1, b])

        # skip reseting agent

        # wait for envs to finish reset
        for b, env in enumerate(self.envs):
            env.await_step()

    def time_batches(self):
        batch_T, batch_B = self.batch_spec
        durations = self.durations

        action = self.batch_buffer.agent.action
        observation, reward, done, env_info = (
            self.batch_buffer.env.observation,
            self.batch_buffer.env.reward,
            self.batch_buffer.env.done,
            self.batch_buffer.env.env_info,
        )

        if self.profiler is not None:
            self.profiler.enable()

        for _ in range(self.n_iterations):
            for t in range(batch_T):
                # all environments receive the same actions
                action[t] = [self.action_space.sample()] * batch_B

                # don't include action space sampling in benchmark
                start = time.perf_counter()

                for b, env in enumerate(self.envs):
                    env.step_async(action[t, b],
                        out_obs=observation[t+1, b], out_reward=reward[t, b],
                        out_done=done[t, b], out_info=env_info[t, b])

                for b, env in enumerate(self.envs):
                    env.await_step()

                end = time.perf_counter()
                durations[t] = (end - start)

            print(f"Average step duration {durations.mean()*1000/batch_B:.4f} "
                    f"+/- {durations.std()*1000/batch_B:.4f} (ms) "
                    f"[{batch_B/durations.mean():.2f} FPS]")

        if self.profiler is not None:
            self.profiler.disable()
            self.profiler.dump_stats(str(self.profile_path))

        return durations