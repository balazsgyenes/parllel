import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb

# isort: split
from build_sb3_ppo import build


@hydra.main(version_base=None, config_path="conf", config_name="ppo_cheetah_sb3")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["ppo", "halfcheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    with open_dict(config):
        config["log_dir"] = run.dir  # algo needs to know where to save tensorboard logs

    with build(config) as (model, learn_kwargs):
        model.learn(**learn_kwargs)

    run.finish()


if __name__ == "__main__":
    main()
