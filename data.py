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
from transform import train_transform

class MyDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        dataset_name_or_path: str = "",
        task_prompt: str = "<123>",
    ):
        super().__init__()
        self.split = split
        self.task_prompt = task_prompt

        # 远程下载
        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)

        # 下载在本地
        self.dataset = load_from_disk(dataset_name_or_path)
        if self.split == "train":
            self.dataset = self.dataset.filter(lambda x: x['split'] == 'train').select(range(20000))
        else:
            self.dataset = self.dataset.filter(lambda x: x['split'] == 'test').select(range(80))
        
        self.dataset_length = len(self.dataset)
        
        # 获取测试集
        print(f"{self.split}: {self.dataset_length}\n{dataset_name_or_path}")


    def __len__(self) -> int:
        return self.dataset_length


    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]
        
        return image, question, answer


if __name__ == "__main__":
    get_max_min_size_of_image()