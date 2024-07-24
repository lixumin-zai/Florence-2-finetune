# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:44:06
# @Author  :   lixumin1030@gmail.com
# @FileName:   inference.py


from transformers import AutoModelForCausalLM, AutoProcessor
import requests
import torch
from transform import test_transform
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import feishu_sdk
from feishu_sdk.sheet import FeishuSheet, FeishuImage
from io import BytesIO
import json
import os
from lightning_module import ModelPLModule
from PIL import ImageFont, Image, ImageDraw, ImageOps
import time
import tqdm

# 加载预训练的模型
model_path = ""
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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
    generated_text = processor.batch_decode(preds, skip_special_tokens=False)
    print(generated_text)

def draw_box(image, data):
    bboxes, labels = data["<OD>"]["bboxes"], data["<OD>"]["labels"]
    draw = ImageDraw.Draw(image)
    for box, label in zip(bboxes, labels):
        box = [int(i) for i in box]
        rectangle_color = (255, 0, 0)
        draw.rectangle(box, outline=rectangle_color)
        text_color = (0, 0, 255)  # 蓝色
        font = ImageFont.truetype('./NotoSansSC-Regular.otf', size=10)
        # 在图片上添加文本
        draw.text((box[2], box[3]), label, fill=text_color, font=font)
    temp_data = BytesIO()
    image.save(temp_data, format="JPEG")
    new_image_bytes = temp_data.getvalue()
    return new_image_bytes

def inference_feishu():
    app_id, app_key = "", ""
    sheet_token, sheet_id = "", ""
    feishu_sdk.login(app_id, app_key)
    sheet = FeishuSheet(sheet_token, sheet_id)
    
    idx = 3
    image_col = "W"
    result_col = "JA"

    prompt = ["<OD>"]

    for i in range(min(sheet.rows+1, 10000)):
        if i < idx:
            continue 
        if sheet[f"{result_col}{i}"]:
            continue
        image_bytes = sheet[f"{image_col}{i}"].image_bytes
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        padding = (20, 20, 20, 20) 
        image = ImageOps.expand(image, padding, fill=(255, 255, 255))
        inputs = processor(text="Detecting all points of geometric shape.", images=image, return_tensors="pt").to("cuda")
        inputs = processor(text="What angles are marked with values in the figure?", images=image, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                do_sample=False,
                num_beams=1
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # object det
        #parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
        #new_image = draw_box(image, parsed_answer)
        #sheet[f"{result_col}{i}"] = FeishuImage(img_bytes=new_image)
        #sheet[f"Z{i}"] = json.dumps(parsed_answer)

        # VQA
        parsed_answer = processor.post_process_generation(generated_text, task="<VIE>", image_size=(image.width, image.height))
        sheet[f"{result_col}{i}"] = json.dumps(parsed_answer, ensure_ascii=False)

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