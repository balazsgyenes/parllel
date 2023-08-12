from envs.cartpole import build_cartpole
from gymnasium import spaces
from models import ActorCriticModel

from parllel.cages import TrajInfo
from parllel.types import BatchSpec
from parllel.patterns import build_cages_and_sample_tree
from agent import JaxAgent


def build():
    batch_spec = BatchSpec(10, 8)
    TrajInfo.set_discount(0.99)

    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=build_cartpole,
        env_kwargs={"max_episode_steps": 1000},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=False,
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    assert isinstance(obs_space, spaces.Box)
    assert isinstance(action_space, spaces.Discrete)

    model = ActorCriticModel(
        actor_hidden_sizes=[256, 256],
        critic_hidden_sizes=[256, 256],
        action_dim=action_space.n,
    )

    agent = JaxAgent(model, metadata.example_obs_batch)

if __name__ == "__main__":
    build()
