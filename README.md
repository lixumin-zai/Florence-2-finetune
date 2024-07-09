
## Florence-2 finetune with pytorch_lightning
latex-ocr

#### env
```
pytorch-lightning==2.3.2
transformers==4.41.2
sconf==1.14.0
tensorboardX==2.6.2.2
datasets==2.20.0
timm==1.0.7
einops==0.8.0
albumentations==1.4.7
timm==1.0.7
nltk==3.8.1
```
```
# 2.3.1
pip install torch torchvision torchaudio

# https://github.com/Dao-AILab/flash-attention/releases
# 2.5.9.post1 flash_attn-2.5.9.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.9.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```

#### prepare
https://huggingface.co/microsoft/Florence-2-base-ft

OleehyO/latex-formulas

#### run
```
python train.py
or
nohup python train.py > log.log &
```


#### attention
```
# Modify the config.json file to inference
- vision_config
    - model_type == davit
```