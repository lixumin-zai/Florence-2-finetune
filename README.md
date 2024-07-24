
## Florence-2 finetune with pytorch_lightning

#### env
```
conda create florence python=3.10
```
```
pytorch-lightning==2.3.2
transformers==4.41.2
sconf==0.2.5
tensorboardX==2.6.2.2
datasets==2.20.0
timm==1.0.7
einops==0.8.0
albumentations==1.4.7
timm==1.0.7
nltk==3.8.1
```
```
# pytorch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# https://github.com/Dao-AILab/flash-attention/releases
# 2.5.9.post1 flash_attn-2.5.9.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.9.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```

#### 不支持使用 flash_attn 需要修改 2 部分
- https://huggingface.co/microsoft/Florence-2-base-ft 中 modeling_florence2.py
```

```

- transformers 中的判断
``` python
# venv/lib/python3.10/site-packages/transformers/dynamic_module_utils.py 中 check_imports

# 添加
if "flash_attn" in imports:
    imports.remove("flash_attn")
```


#### run
```
python train.py
or
nohup python train.py > log.log &
```
#### prepare
https://huggingface.co/microsoft/Florence-2-base-ft


#### attention
```
# cudnn problem
export LD_LIBRARY_PATH=/store/lixumin/venv/florence/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH


# Modify the config.json file to inference
- vision_config
    - model_type == davit
```

