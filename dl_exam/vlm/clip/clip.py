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
from .tokenizer import simple_tokenizer as _Totokenizer
