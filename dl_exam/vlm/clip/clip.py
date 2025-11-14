"""
Author: Redal
Date: 2025-11-04
Todo: clip.py to download pretrained model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import urllib
import warnings
import hashlib
from tqdm import tqdm
from packaging import version
from typing import Union, List, Tuple
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize
from torchvision.transforms import CenterCrop, ToTensor
from .model import build_model
from .tokenizer import Tokenizer as _Totokenizer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("Pytorch的版本需要更新至1.7.1以上")


__all__ = ["available_models", "load", "tokenize"]
__tokenizer__ = _Totokenizer()
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",}


def _download(url:str, root:str):
    """下载预训练的模型权重"""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    # 提取sha256编码用于校验后续文件完整性
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f'{download_target}目前存在并且不是正常的文件')
    # SHA256文件校验码对比检验
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, 'rb').read()).hexdigest()==expected_sha256:
            return download_target
        else: warnings.warn(f'{download_target}文件存在,但是SHA256编码不匹配,需删除重新下载')
    # 下载预训练权重文件
    # urllib.request.urlopen(url)：创建一个网络连接对象，用于从指定的url获取数据
    # open(download_target, 'wb')：以二进制写入模式打开本地文件，准备保存下载的内容
    with urllib.request.urlopen(url) as source, open(download_target, 'wb') as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer: break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, 'rb').read()).hexdigest() != expected_sha256:
        raise RuntimeError("预训练模型权重已经被下载,但是SHA256编码不匹配,需要重新下载")
    return download_target
        

def _convert_image_to_rgb(image):
    