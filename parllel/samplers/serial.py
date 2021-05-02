from nptyping import NDArray

from .types import Samples

class Sampler:
    def __init__(self,
        batch_T: int,
        batch_B: int,
        get_bootstrap_value: bool = False,
    ) -> None:
        self.batch_T = batch_T
        self.batch_B = batch_B
        self.get_bootstrap_value = get_bootstrap_value

        self.env_samples = None  # namedtuplearray to store env samples
        self.agent_samples = None  # namedtuplearray to store agent samples
        self.batch_observation = None  # array to store last observations, written by envs, read by agent
        self.batch_action = None  # array to store last actions, written by agent, read by envs
        self._last_iteration = None  # placeholder for previous values stored from last iteration

    def initialize(self, agent, envs, batch_buffer: Samples) -> None:
        self.agent = agent
        self.envs = envs
        assert len(envs) == self.batch_B

        self.batch_buffer = batch_buffer

    def collect_batch(self, elapsed_steps) -> Samples:
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        observation, action, reward = self._last_iteration
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(elapsed_steps)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            action, agent_info, next_rnn_state = self.agent.step(observation, action, reward, rnn_state)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                if d:
                    o = env.reset()
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(observation, action, reward)

    def shutdown(self):
        pass