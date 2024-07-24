# -*- coding: utf-8 -*-
# @Time    :   2024/07/09 11:43:39
# @Author  :   lixumin1030@gmail.com
# @FileName:   transform.py


import albumentations as alb
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageFilter
import random
import string
import os
import io

FONT_PATH = ""

def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]
    return f

def Image2cv2(pil_image):
    """ 将 PIL 图像转换为 OpenCV 图像 """
    # 将 PIL 图像转换为 numpy 数组
    cv2_image = np.array(pil_image)
    # 转换颜色通道从 RGB 到 BGR
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv22Image(cv2_image):
    """ 将 OpenCV 图像转换为 PIL 图像 """
    # 转换颜色通道从 BGR 到 RGB
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # 将 numpy 数组转换为 PIL 图像
    pil_image = Image.fromarray(cv2_image)
    return pil_image

def bytes2Image(image_bytes):
    # 使用 io.BytesIO 将字节数据转为一个字节流对象
    image_stream = io.BytesIO(image_bytes)
    
    # 使用 PIL.Image.open 从字节流对象中打开图像
    image = Image.open(image_stream)
    
    return image

def cv2ImageToBytes(image, format='.jpg'):
    """
    将 OpenCV 图像转换为字节数据
    :param image: OpenCV 图像
    :param format: 图像格式（默认为 .jpg）
    :return: 图像的字节数据
    """
    # 使用 imencode 将图像编码为指定格式
    success, encoded_image = cv2.imencode(format, image)
    if not success:
        raise ValueError("Could not encode image")
    
    # 将编码后的图像转换为字节数据
    return encoded_image.tobytes()
################################################################################################

class ResizeIfNeeded(alb.ImageOnlyTransform):
    def __init__(self, max_size, min_size, always_apply=False, p=1.0):
        super(ResizeIfNeeded, self).__init__(always_apply, p)
        self.max_size = max_size
        self.min_size = min_size

    def apply(self, img, **params):
        # 获取图片的高度和宽度
        # img = simulate_color_jitter(img)
        height, width = img.shape[:2]
        # 获取最长边和最短边
        max_side = max(height, width)
        min_side = min(height, width)
        
        # 如果最长边超过 max_size，则等比例缩放
        if max_side > self.max_size:
            scale = self.max_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width  # 更新高度和宽度

        # 如果最短边小于 min_size，则等比例缩放
        min_side = min(height, width)  # 更新后的最短边
        max_side = max(height, width)
        if max_side < self.min_size:
            scale = self.min_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img

################################################################################################

class ResizeIfMaxSideExceeds(alb.ImageOnlyTransform):
    def __init__(self, max_size, always_apply=False, p=1.0):
        super(ResizeIfMaxSideExceeds, self).__init__(always_apply, p)
        self.max_size = max_size

    def apply(self, img, **params):
        # 获取图片的高度和宽度
        height, width = img.shape[:2]
        # 获取最长边
        max_side = max(height, width)
        
        if max_side > self.max_size:
            # 计算缩放比例
            scale = self.max_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            # 等比例缩放图片
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
################################################################################################

class ConditionalRandomScale(alb.ImageOnlyTransform):
    def __init__(self, scale_limit=(-0.5, 1), enlarge=False, always_apply=False, p=1.0):
        super(ConditionalRandomScale, self).__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.enlarge = enlarge

    def apply(self, img, **params):
        # 如果图像尺寸大于或等于 192x192，执行 RandomScale
        if self.enlarge:
            return alb.RandomScale(scale_limit=self.scale_limit, p=1)(image=img)['image']
        else:
            pass

        if img.shape[0] >= 320 and img.shape[1] >= 320:
            return alb.RandomScale(scale_limit=self.scale_limit, p=1)(image=img)['image']
        else:
            return img
        
################################################################################################

class Line_blur(alb.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Line_blur, self).__init__(always_apply, p)
        

    def apply(self, img, **params):
        img = cv22Image(img)
        shadow = Image.new('RGBA', img.size, (0, 0, 0, 20))
        offset = (0, 0)
        img.paste(shadow, offset, shadow)
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        # 将字节流转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        # 使用cv2.imdecode将numpy数组转换为cv2图像
        cv2_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        return cv2_img

################################################################################################


def apply_watermark(src_image, text, text_size, rotation_angle):
    # Load the original image
    # Create a single watermark
    watermark = np.zeros((random.randint(60, 200), random.randint(100, 400), 3), dtype=np.uint8)*255
    r, g, b = random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)
    watermark = put_text_husky(watermark, text, (r, g, b), text_size, "Times New Roman")

    # Define horizontal and vertical repeat counts based on the size of the source image
    h_repeat = src_image.shape[1] // watermark.shape[1] + 1
    v_repeat = src_image.shape[0] // watermark.shape[0] + 1

    # Create tiled watermark
    tiled_watermark = np.tile(watermark, (v_repeat, h_repeat, 1))

    # Crop the tiled watermark to the size of the original image
    tiled_watermark = tiled_watermark[:src_image.shape[0], :src_image.shape[1]]
    # Rotate the watermark
    center = (tiled_watermark.shape[1] // 2, tiled_watermark.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_watermark = cv2.warpAffine(tiled_watermark, rotation_matrix, (tiled_watermark.shape[1], tiled_watermark.shape[0]))
    # src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2BGRA)
    
    # src_image = cv22Image(src_image)
    # rotated_watermark = cv22Image(rotated_watermark)

    image = cv2.addWeighted(src_image, 0.8, rotated_watermark, random.uniform(0.2, 0.5), 1)
    # image = np.where(image == (242, 242, 242), 255, image)
    return image

def put_text_husky(img, text, color, font_size, font_name, italic=False, underline=False):
    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Set font style
    font_style = ''
    if italic:
        font_style += 'I'
    if underline:
        font_style += 'U'
    
    # Load font or default
    try:
        # font = ImageFont.truetype(f'{font_name}{font_style}.ttf', font_size)
        font_name = os.listdir(FONT_PATH)
        font = ImageFont.truetype(f"{FONT_PATH}/{random.choice(font_name)}", font_size)

    except IOError:
        print(f"Font {font_name} with style {font_style} not found. Using default font.")
        font = ImageFont.load_default()

    # Calculate text bounding box for center alignment
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    orgX = (img.shape[1] - text_width) // 2
    orgY = (img.shape[0] - text_height) // 2

    # Draw text
    
    draw.text((orgX, orgY), text, font=font, fill=(int(color[0]), int(color[1]), int(color[2])))
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

import random

def generate_random_chinese_char():
    # 生成一个随机的汉字Unicode编码值
    unicode_val = random.randint(0x4E00, 0x9FFF)
    # 将Unicode编码值转换为对应的字符
    return chr(unicode_val)

def generate_balanced_watermark_text(length=15):
    # 定义可能的字符池
    chinese_chars = "".join([generate_random_chinese_char() for i in range(20)])
    english_chars = string.ascii_uppercase  # A-Z
    digits = string.digits  # 0-9
    
    # 确保每种类型的字符至少出现一次
    if length < 3:
        raise ValueError("Length must be at least 3 to include at least one of each character type.")
    
    # 生成包含至少一个中文、一个英文字母和一个数字的基础水印文本
    watermark_text = [
        random.choice(chinese_chars),
        random.choice(english_chars),
        random.choice(digits)
    ]
    
    # 填充剩余的字符
    all_chars = chinese_chars + english_chars + digits
    watermark_text += [random.choice(all_chars) for _ in range(length - 3)]
    
    # 混洗以增加随机性
    random.shuffle(watermark_text)
    
    # 将列表转换为字符串
    return ''.join(watermark_text)

class watermark(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        random_watermark_text = generate_balanced_watermark_text(random.randint(3, 7))  # 生成15个字符的水印文本
        # print(random_watermark)
        font_size = random.randint(25, 100)
        rotation_angle = random.randint(-50, 50)
        # Example usage:
        result_img = apply_watermark(img, random_watermark_text, font_size, rotation_angle)
        return result_img

################################################################################################
def generate_random_rgb():
    r = random.randint(0, 20)
    g = random.randint(0, 20)
    b = random.randint(0, 20)
    return (r, g, b)

def add_random_shadows(input_image):
    # 将OpenCV图像转换为PIL图像
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(input_image)

    width, height = pil_image.size

    # 创建阴影图层
    shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    # 随机生成阴影形状
    opacity = random.randint(50, 170) # 不透明度
    for _ in range(random.randint(5, 10)):  # 生成5-10个随机形状
        # 随机选择形状类型：圆形或椭圆形
        shape_type = random.choice(['ellipse', 'ellipse', 'ellipse', 'ellipse', 'rectangle'])
        # 随机生成位置和大小
        x0 = random.randint(0, int(width / 1.3))
        y0 = random.randint(0, int(height / 1.3))
        x1 = random.randint(int(1.2*x0), width)
        y1 = random.randint(int(1.2*y0), height)
        # 随机生成透明度
        # 绘制形状
        fill_data = generate_random_rgb() + tuple([opacity])
        if shape_type == 'ellipse':
            shadow_draw.ellipse([x0, y0, x1, y1], fill=fill_data)
        else:
            shadow_draw.rectangle([x0, y0, x1, y1], fill=fill_data)

    # 使用高斯模糊使阴影更自然
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))

    # 将阴影合并到原图上
    pil_image.paste(shadow, mask=shadow)

    # 将PIL图像转换回OpenCV图像
    result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return result_image

class add_shadown(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # 一个水印
        new_img = add_random_shadows(img)  # 生成15个字符的水印文本
        return new_img
    
################################################################################################

train_transform =  alb_wrapper(
    alb.Compose(
        [
            ConditionalRandomScale(scale_limit=(-0.5, -0.3), p=1),
            add_shadown(p=0.5),
            alb.GaussNoise(20, p=0.8),
            alb.GaussianBlur((3, 3), p=0.8),
            Line_blur(p=0.95),
            watermark(p=0.1),
            alb.Blur(blur_limit=7, p=1),
            alb.MotionBlur(blur_limit=(7, 13), p=0.3),  # 动态模糊
            alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.8),
            alb.RandomBrightnessContrast(p=0.8),  # p=1.0表示总是应用这个变换
            alb.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.8),
        ],
    )
)
test_transform =  alb_wrapper(
    alb.Compose(
        [
            ResizeIfMaxSideExceeds(max_size=384),  # 首先等比例缩放图像的最长边为1024
            alb.PadIfNeeded(min_height=480, min_width=480, border_mode=0, value=(255, 255, 255)),  # 然后添加必要的填充到1024x1024，使用边界模式0（常数填充）
        ],
    )
)

if __name__ == '__main__':
    root_path = "./img/"
    image_name = os.listdir(root_path)
    for i in range(10):
        # print(1)
        image_path = root_path + random.choice(image_name)
        image = Image.open(image_path).convert('RGB')
        image = Image.fromarray(train_transform(image))
        image.save(f"../show_{i}.jpg")