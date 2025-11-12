"""
Author: Redal
Date: 2025-11-04
Todo: __init__.py for vlm tasks
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
from torch.nn import funcutional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False
try:
    import hotrovod.torch as hvd
except ImportError:
    hvd = None

def gathor_features(
        image_features.
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False):
    assert has_distributed, "Error"
    if use_horovod:
        assert hvd is not None, "Error"
        