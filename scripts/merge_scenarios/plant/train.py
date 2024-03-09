from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import torch


from src.carla.config.config import DefaultMainConfig
from src.carla.config.observation_configs import EntityObservationConfig
from src.carla.config.action_configs import ThrottleConfig
from src.carla.config.scenario_configs import DefaultScenarioConfig
from src.carla.features.scenario_datasets import ScenarioH5Dataset, scenario_collate_fn
from src.plant.lit_module import LitHFLM


@hydra.main(version_base=None, config_path='conf/', config_name='train.yaml')
def main(cfg):

    # print config
    print(OmegaConf.to_yaml(cfg))

    # setup logging
    seed_everything(cfg.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = DefaultMainConfig()
    config.populate_config(
        observation_config=EntityObservationConfig(max_route_pts=70, max_actors=30, max_token_distance=50.),
        action_config=ThrottleConfig(),
        scenario_config=DefaultScenarioConfig(),
        testing=False,
    )

    train_dataset = ScenarioH5Dataset(config=config, **cfg.train_dataset)

    model = LitHFLM(cfg=cfg)

    callbacks = []

    if cfg.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=cfg.checkpoint_freq,
            dirpath=cfg.model_savedir,
        )
        callbacks.append(checkpoint_callback)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers>0),
        collate_fn=scenario_collate_fn
    )

    val_dataset = ScenarioH5Dataset(config=config, **cfg.val_dataset)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers>0),
        collate_fn=scenario_collate_fn
    )

    trainer = pl.Trainer(
        logger=None,
        callbacks=callbacks,
        max_epochs=cfg.num_epochs,
        **cfg.trainer,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    print('Done')


if __name__ == '__main__':
    main()
