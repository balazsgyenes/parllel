from dataclasses import dataclass


@dataclass
class SB3EvalTrajInfo:
    mean_ep_length: int = 0
    mean_reward: float = 0

    def step(
        self,
        observation,
        action,
        reward,
        terminated,
        truncated,
        env_info,
    ) -> None:
        self.mean_ep_length += 1
        self.mean_reward += reward


@dataclass
class SB3TrajInfo:
    ep_len_mean: int = 0
    ep_rew_mean: float = 0

    def step(
        self,
        observation,
        action,
        reward,
        terminated,
        truncated,
        env_info,
    ) -> None:
        self.ep_len_mean += 1
        self.ep_rew_mean += reward
