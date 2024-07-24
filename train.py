# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:43:26
# @Author  :   lixumin1030@gmail.com
# @FileName:   train.py


import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config
import json

from data import MyDataset
from lightning_module import DataPLModule, ModelPLModule

@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")

class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.enable = True
        self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            items["lr"] = f"{current_lr:.5e}"  # 格式化学习率显示
        items["name"] = f"{self.config.get('name', '')}"
        items["version"] = f"{self.config.get('version', '')}"
        return items


def set_seed(seed):
    pl.seed_everything(seed)


def train(config):
    set_seed(config.get("seed", 42))

    model_module = ModelPLModule(config)
    data_module = DataPLModule(config)

    logger = TensorBoardLogger(
        save_dir=config.save_path,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    # 按照 ckpt 保存
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=Path(config.save_path),
    #     filename="artifacts-{epoch}",  # 文件名包含 epoch 编号
    #     save_top_k=-1,  # 保存所有 epochs 的 checkpoints
    #     every_n_epochs=1,  # 每个 epoch 保存一次
    #     save_last=False,  # 也保存最后一个 epoch 的模型
    #     verbose=True,  # 打印保存信息
    # )

    # 按照 step 保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.save_path),
        filename="artifacts--{epoch:02d}-{step:05d}",  # 文件名包含 step 编号
        save_top_k=-1,  # 保存所有 checkpoints
        save_last=True,  # 保存最后一个步骤的模型
        verbose=True,  # 打印保存信息
        every_n_train_steps=config.save_step,  # 每 100 训练步骤保存一次
        save_on_train_epoch_end=False,  # 防止在每个 epoch 结束时额外保存
    )

    bar = ProgressBar(config)

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        # devices=[3],
        devices=config.devices,
        strategy="ddp",
        accelerator="gpu",
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback, bar],
    )
    try:
        trainer.fit(model_module, data_module, 
            ckpt_path=None)
    except:
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint("trainer_last.ckpt")
        print('trainer_last.ckpt saved.')
    

if __name__ == "__main__":
    config_path = "./config.yaml"
    config = Config(config_path)
    save_config_file(config, Path(config.save_path))
    train(config)
