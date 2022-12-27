from contextlib import contextmanager
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import torch

from parllel.arrays import (Array, RotatingArray, SharedMemoryArray, 
    RotatingSharedMemoryArray, buffer_from_example)
from parllel.buffers import AgentSamples, buffer_method, Samples
from parllel.cages import TrajInfo
from parllel.configuration import add_default_config_fields
import parllel.logger as logger
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
    add_reward_clipping, add_reward_normalization, add_valid,
    build_cages_and_env_buffers, add_initial_rnn_state)
from parllel.replays import BatchedDataLoader
from parllel.runners.onpolicy import OnPolicyRunner
from parllel.samplers import RecurrentSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import (PPO, add_default_ppo_config,
    build_dataloader_buffer)
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.transforms.video_recorder import RecordVectorizedVideo
from parllel.types import BatchSpec

from hera_gym.builds.visual_cartpole import build_visual_cartpole
from models.atari_lstm_model import AtariLstmPgModel


@contextmanager
def build(config: Dict) -> OnPolicyRunner:

    parallel = config["parallel"]
    batch_spec = BatchSpec(
        config["batch_T"],
        config["batch_B"],
    )
    TrajInfo.set_discount(config["discount"])

    if parallel:
        ArrayCls = SharedMemoryArray
        RotatingArrayCls = RotatingSharedMemoryArray
    else:
        ArrayCls = Array
        RotatingArrayCls = RotatingArray

    cages, batch_action, batch_env = build_cages_and_env_buffers(
        EnvClass=build_visual_cartpole,
        env_kwargs=config["env"],
        TrajInfoClass=TrajInfo,
        reset_automatically=False,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    # instantiate model and agent
    model = AtariLstmPgModel(
        obs_space=obs_space,
        action_space=action_space,
        **config["model"],
    )
    distribution = Categorical(dim=action_space.n)
    device = torch.device(config["device"])

    # instantiate model and agent
    agent = CategoricalPgAgent(
        model=model,
        distribution=distribution,
        observation_space=obs_space,
        action_space=action_space,
        n_states=batch_spec.B,
        recurrent=True,
    )
    agent = TorchHandler(agent=agent)

    # write dict into namedarraytuple and read it back out. this ensures the
    # example is in a standard format (i.e. namedarraytuple).
    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]
    
    # get example output from agent
    _, agent_info = agent.step(example_obs)

    # allocate batch buffer based on examples
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), ArrayCls)
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    # for recurrent problems, we need to save the initial state at the 
    # beginning of the batch
    batch_buffer = add_initial_rnn_state(batch_buffer, agent)

    # for advantage estimation, we need to estimate the value of the last
    # state in the batch
    batch_buffer = add_bootstrap_value(batch_buffer)
    
    # for recurrent problems, compute mask that zeroes out samples after
    # environments are done before they can be reset
    batch_buffer = add_valid(batch_buffer)

    # add several helpful transforms
    batch_transforms = []

    batch_buffer, batch_transforms = add_reward_normalization(
        batch_buffer,
        batch_transforms,
        discount=config["discount"],
    )

    batch_buffer, batch_transforms = add_reward_clipping(
        batch_buffer,
        batch_transforms,
        reward_clip_min=config["reward_clip_min"],
        reward_clip_max=config["reward_clip_max"],
    )

    # add advantage normalization, required for PPO
    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=config["discount"],
        gae_lambda=config["gae_lambda"],
        normalize=config["normalize_advantage"],
    )

    # TODO: log videos without calling `log`, which causes tensorboard to go out of sync
    if video_config := config.get("video_recorder", {}):
        video_recorder = RecordVectorizedVideo(
            batch_buffer=batch_buffer,
            buffer_key_to_record="observation",
            env_fps=50, # TODO: grab this from example env metadata
            **video_config,
        )
        batch_transforms.append(video_recorder)

    sampler = RecurrentSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_dataloader_buffer(batch_buffer, recurrent=True)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
        sampler_batch_spec=batch_spec,
        n_batches=config["minibatches"],
        batch_only_fields=["init_rnn_state"],
        recurrent=True,
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config["learning_rate"],
        **config["optimizer"],
    )
    
    # create algorithm
    algorithm = PPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        **config["algo"],
    )

    # create runner
    runner = OnPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        **config["runner"],
    )

    try:
        yield runner
    
    finally:
        sampler.close()
        if video_config := config.get("video_recorder", {}):
            video_recorder.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_buffer, "close")
        buffer_method(batch_buffer, "destroy")
    

if __name__ == "__main__":
    mp.set_start_method("fork")

    config = dict(
        parallel=False,
        batch_T=128,
        batch_B=16,
        discount=0.99,
        learning_rate=0.001,
        gae_lambda=0.95,
        reward_clip_min=-5,
        reward_clip_max=5,
        normalize_advantage=True,
        max_steps_decorrelate=50,
        render_during_training=False,
        env=dict(
            max_episode_steps=1000,
            reward_type="sparse",
            headless=True,
        ),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model=dict(
            channels=[32, 64, 128, 256],
            kernel_sizes=[3, 3, 3, 3],
            strides=[2, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            use_maxpool=False,
            post_conv_hidden_sizes=1024,
            post_conv_output_size=None,
            post_conv_nonlinearity=torch.nn.ReLU,
            lstm_size=512,
            post_lstm_hidden_sizes=512,
            post_lstm_nonlinearity=torch.nn.ReLU,
        ),
        runner=dict(
            n_steps=200 * 16 * 128,
            log_interval_steps=1e4,
        ),
    )

    if config.get("render_during_training", False):
        config["env"]["headless"] = False
        config["env"]["subprocess"] = config["parallel"]

    config = add_default_ppo_config(config)
    config = add_default_config_fields(config)

    # default values if wandb is not installed (or not used)
    run = None
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M')

    try:
        import wandb
        run = wandb.init(
            anonymous="must", # for this example, send to wandb dummy account
            project="CartPole",
            group="PPO",
            tags=["discrete", "image-based", "ppo"],
            config=config,
            sync_tensorboard=True,  # auto-upload any values logged to tensorboard
            save_code=True,  # save script used to start training, git commit, and patch
        )
        run_id = run.id
    except ImportError:
        pass

    config["video_recorder"] = dict(
        record_every_n_steps=5e4,
        video_length=250,
        output_dir=Path(f"videos/{run_id}"),
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/cartpole-visual-ppo/{run_id}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=config,
        model_save_path="model.pt",
        # verbosity=Verbosity.DEBUG,
    )

    with build(config) as runner:
        runner.run()

    if run is not None:
        run.finish()
