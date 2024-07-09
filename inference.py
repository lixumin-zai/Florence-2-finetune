# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:44:06
# @Author  :   lixumin1030@gmail.com
# @FileName:   inference.py


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderConfig
from PIL import Image
import requests
import torch
from transform import test_transform
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
from io import BytesIO
from utils import JsonParse
import json
import os
from lightning_module import GeoVIEModelPLModule

# 加载预训练的模型
model = AutoModelForCausalLM.from_pretrained("/disk1/xizhi/cv/lixumin/Florence-2-base", trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained("/disk1/xizhi/cv/lixumin/Florence-2-base", trust_remote_code=True)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def test_one():
    image_path = ""
    image = Image.open(image_path).convert("RGB")
    prompt = "Recognize latex text in the image."

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    # 预热
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3
        )

    import time
    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3
        )
    generated_text = processor.batch_decode(preds, skip_special_tokens=False)
    print(time.time() - st)
    print(generated_text)


def inference_feishu():
    feishu_sdk.login("", "")
    sheet_token, sheet_id = "KXaqs0sI9hH7YxtvhO0cpgfmnim", "f4c320"
    sheet = FeishuSheet(sheet_token, sheet_id)
    
    idx = 3
    image_col = "W"
    result_start_col = "IG"
    result_end_col = "IO"

    prompt = ["<123>"]

    for i in range(min(sheet.rows+1, 10000)):
        if i < idx:
            continue 
        # if sheet[f"{result_start_col}{i}"]:
        #     continue
        image_bytes = sheet[f"{image_col}{i}"].image_bytes
        inputs = processor(text=prompt, images=Image.open(BytesIO(image_bytes)).convert("RGB"), return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task="<123>", image_size=(image.width, image.height))
        # sheet[f"{result_start_col}{i}": f"{result_end_col}{i}"] = [
        #     result[:-1]
        # ]
        sheet[f"{result_start_col}{i}"] = str(parsed_answer)
        # break

def save_ckpt():
    from lightning_module import ModelPLModule
    import pytorch_lightning as pl
    from sconf import Config
    model_path = ""
    save_path = ""
    ckpt_name = ""
    os.path.exists(save_path) or os.makedirs(save_path)
    config = Config(f"{model_path}/config.yaml")
    config.save_path = save_path
    config.argv_update()
    pl.seed_everything(config.get("seed", 42))
    config.pretrained_model_name_or_path = model_path
    pretrained_model = ModelPLModule.load_from_checkpoint(model_path+ckpt_name, config=config)
    pretrained_model.on_save_checkpoint()

if __name__ == "__main__":
    # test_one()
    # save_ckpt()
    inference_feishu()