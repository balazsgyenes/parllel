from __future__ import annotations

import cProfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import parllel.logger as logger
from parllel import Array, ArrayDict
from parllel.cages import Cage
from parllel.types import BatchSpec

from .sampler import Sampler


class ProfilingSampler(Sampler):
    def __init__(
        self,
        batch_spec: BatchSpec,
        envs: Sequence[Cage],
        sample_tree: ArrayDict[Array],
        n_iterations: int,
        profile_path: Path | None = None,
    ) -> None:
        super().__init__(
            batch_spec=batch_spec,
            envs=envs,
            agent=None,
            sample_tree=sample_tree,
        )

        self.n_iterations = n_iterations
        self.profile_path = profile_path

        self.action_space = envs[0].spaces.action
        self.durations = np.zeros((batch_spec.T,), dtype=float)
        self.profiler = cProfile.Profile() if profile_path is not None else None

    def reset_agent(self) -> None:
        # skip resetting agent
        pass

    def collect_batch(self) -> np.ndarray:
        batch_T, batch_B = self.batch_spec
        durations = self.durations

        action = self.sample_tree["action"]
        observation = self.sample_tree["observation"]
        reward = self.sample_tree["reward"]
        done = self.sample_tree["done"]
        terminated = self.sample_tree["terminated"]
        truncated = self.sample_tree["truncated"]
        env_info = self.sample_tree["env_info"]

        if self.profiler is not None:
            self.profiler.enable()

        for _ in range(self.n_iterations):
            for t in range(batch_T):
                for b in range(batch_B):
                    action[t, b] = self.action_space.sample()

                # don't include action space sampling in benchmark
                start = time.perf_counter()

                for b, env in enumerate(self.envs):
                    env.step_async(
                        action[t, b],
                        out_obs=observation[t + 1, b],
                        out_reward=reward[t, b],
                        out_terminated=terminated[t, b],
                        out_truncated=truncated[t, b],
                        out_info=env_info[t, b],
                    )

                for b, env in enumerate(self.envs):
                    env.await_step()

                done[t] = np.logical_or(terminated[t], truncated[t])

                end = time.perf_counter()
                durations[t] = end - start

            logger.info(
                f"Average step duration {durations.mean()*1000/batch_B:.4f} "
                f"+/- {durations.std()*1000/batch_B:.4f} (ms) "
                f"[{batch_B/durations.mean():.2f} FPS]"
            )

        if self.profiler is not None:
            self.profiler.disable()
            self.profiler.dump_stats(str(self.profile_path))

        return durations
