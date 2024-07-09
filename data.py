# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:44:13
# @Author  :   lixumin1030@gmail.com
# @FileName:   data.py


import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import PIL
import os
import tqdm

from typing import Any, List, Optional, Union
import json
from torchvision import transforms
import random


class MyDataset(Dataset):
    def __init__(
        self,
        processor,
        split: str = "train",
        dataset_name_or_path: str = "",
        task_prompt: str = "",
    ):
        super().__init__()
        self.split = split
        self.task_prompt = "<123>"
        self.processor = processor

        # 远程加载
        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)

        # 本地加载
        if self.split == "train":
            self.dataset = load_from_disk(dataset_name_or_path)["train"]
        else:
            self.dataset = load_from_disk(dataset_name_or_path)["train"].select(range(200))
        self.dataset_length = len(self.dataset)

        print(f"{self.split}: {self.dataset_length}\n{dataset_name_or_path}")


    def __len__(self) -> int:
        return self.dataset_length


    def __getitem__(self, idx):
        # 为了适应公式数据
        sample = self.dataset[idx]
        images = sample["image"]
        question = "Recognize latex text in the image."
        answer = sample["latex_formula"]

        inputs = self.processor(
            text=[question], images=[images], return_tensors="pt", padding=True
        )
        input_ids, pixel_values = inputs["input_ids"].squeeze(), inputs["pixel_values"].squeeze()

        labels = self.processor.tokenizer(
            text=answer,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        ).input_ids.squeeze()
        if self.split != "train":
            return input_ids, pixel_values, [answer]

        return input_ids, pixel_values, labels


if __name__ == "__main__":
    get_max_min_size_of_image()