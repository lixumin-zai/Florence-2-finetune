# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:43:49
# @Author  :   lixumin1030@gmail.com
# @FileName:   test.py


import requests
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

model = AutoModelForCausalLM.from_pretrained("/disk1/xizhi/cv/lixumin/Florence-2-base", trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained("/disk1/xizhi/cv/lixumin/Florence-2-base", trust_remote_code=True)

prompt = "<OD>"

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

image_path = ""
image = Image.open(image_path).convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    st = time.time()
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    print(time.time() - st)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)
input()

# 数据集测试

# from datasets import load_dataset
# dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas") 

# import datasets
# dataset = datasets.load_from_disk("./dataset/OleehyO-latex-formulas/")

# print(type(dataset))
# print(dataset["train"].select(range(200))[0])