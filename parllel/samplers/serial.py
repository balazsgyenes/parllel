from nptyping import NDArray

class Sampler:
    def __init__(self, agent, cages, batch_T) -> None:
        self.agent = agent
        self.batch_T = batch_T
        self.cages = cages
        self.batch_B = len(cages)

        self.env_samples = None  # namedtuplearray to store env samples
        self.agent_samples = None  # namedtuplearray to store agent samples
        self.batch_observation = None  # array to store last observations, written by envs, read by agent
        self.batch_action = None  # array to store last actions, written by agent, read by envs

    def initialize(self) -> None:
        pass

    def collect_batch(self) -> NDArray:
        observation = self.batch_observation
        action = self.batch_actions

        for t in range(self.batch_T):
            self.env_samples.obs[t] = observation.copy()
            action, agent_info = self.agent.step(observation)
            for b in range(self.batch_B):
                env = self.cages[b]
                obs, rew, done, info = env.step(action[b])

    def shutdown(self):
        pass