import logging
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np


from src.plant.model import HFLM


logger = logging.getLogger(__name__)


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.last_epoch = 0
        self.model = HFLM(self.cfg)

        # Loss functions
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)

    def forward(self, full_obs, return_targets=False):
        return self.model(full_obs, device=self.device, return_targets=return_targets)

    def get_action(self, full_obs):
        obs ={}
        for k in full_obs:
            if 'vehicle' in k:
                obs[k] = full_obs[k][:, -1:]
            elif 'route' in k:
                obs[k] = full_obs[k]
        _, pred_wp, _ = self.forward(obs)
        speed = obs['vehicle_features'][:, -1, 0, 3] * 50.
        steer, throttle, brake = self.model.control_pid(pred_wp, speed)
        if brake:
            gas = -0.25
        else:
            gas = throttle
        return np.array((steer, gas))

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.training)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        full_obs = batch.get_obs(ref_t=0, device=self.device)

        # multitask training
        logits, targets, pred_wp, wp, _ = self(full_obs, return_targets=True)
        loss_wp = F.l1_loss(pred_wp, wp)
        losses_forecast = [
            self.criterion_forecast(logits[i], targets[:, i])
            for i in range(len(logits))
        ]
        loss_forecast = torch.mean(torch.stack(losses_forecast))

        loss_all = (
            1                                                           * loss_wp
            + self.cfg.pre_training.get("forecastLoss_weight", 0) * loss_forecast
        )
        self.log(
            "train/loss_forecast",
            loss_forecast,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.cfg.batch_size,
        )
        self.log(
            "train/loss_wp",
            loss_wp,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.cfg.batch_size,
        )

        return loss_all

    def validation_step(self, batch, batch_idx):
        pass
