import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

# isort: split
from build_sb3_sac import build


@hydra.main(version_base=None, config_path="conf", config_name="sac_hopper_sb3")
def main(config: DictConfig) -> None:
    run = wandb.init(
        project="parllel",
        tags=["sac", "halfcheetah"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
    )

    with build(config) as (model, learn_kwargs):
        model.learn(**learn_kwargs)
        run.finish()  # finish wandb run before tensorboard dir is deleted


if __name__ == "__main__":
    main()
