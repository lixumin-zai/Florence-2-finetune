# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:43:57
# @Author  :   lixumin1030@gmail.com
# @FileName:   lightning_module.py


import math
import random
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance
from pytorch_lightning.utilities import rank_zero_only
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, ViTFeatureExtractor
from torch.optim import Adam, AdamW
from transformers import get_cosine_schedule_with_warmup


class ModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrain_model_path = self.config.pretrain_model_path # "/disk1/xizhi/cv/lixumin/Florence-2-base"
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrain_model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.pretrain_model_path, trust_remote_code=True)

    def training_step(self, batch, batch_idx):
        input_ids, pixel_values, labels  = batch[0]

        loss = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values, 
            labels=labels,
            return_dict=True
            ).loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        self.log('loss', loss, prog_bar=True)
        return loss


    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_step_outputs = [[] for _ in range(self.num_of_loaders)]
        return


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, pixel_values, answers = batch

        preds = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values, 
            max_new_tokens=1024,
            num_beams=3,
        )
        preds = self.processor.batch_decode(preds, skip_special_tokens=False)
        scores = []
        for generated_text, answer in zip(preds, answers):
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task="<123>",
                image_size=(
                    pixel_values.shape[-2],
                    pixel_values.shape[-1],
                ),
            )
            print("GT:", answer)
            print("Pred:", parsed_answer)

        self.validation_step_outputs[dataloader_idx].append(scores)
        return scores


    def on_validation_epoch_end(self):
        assert len(self.validation_step_outputs) == self.num_of_loaders
        cnt = [0] * self.num_of_loaders
        total_metric = [0] * self.num_of_loaders
        val_metric = [0] * self.num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)


    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1.0e-6, weight_decay=0.05)
        # 设置 warmup_steps 和 total_steps
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps, 
            num_training_steps=self.config.num_training_samples_per_epoch*self.config.max_epochs
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',  # 指定更新学习率的间隔是每步还是每个 epoch
            'frequency': 1,
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler_config]

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.config.save_path)
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)


class DataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = list()
        for train_dataset, batch_size in zip(self.train_datasets, self.train_batch_sizes):
            loaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )
            )
        return loaders

    def val_dataloader(self):
        loaders = list()
        for val_dataset, batch_size in zip(self.val_datasets, self.val_batch_sizes):
            loaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    shuffle=False,
                )
            )
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

