"""
Author: Redal
Date: 2025-10-11
Todo: transformer.py
Homepape: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn


# 设置transformer不同尺寸的经验性设定模型参数
MODEL_CONFIGS = {
    'tiny': {
        'num_layers': 2,
        'd_model': 128,
        'num_heads': 2,
        'dff': 512,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'small': {
        'num_layers': 4,
        'd_model': 256,
        'num_heads': 4,
        'dff': 1024,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'medium': {
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'dff': 2048,
        "vocab_size": 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'large': {
        'num_layers': 12,
        'd_model': 1024,
        'num_heads': 16,
        'dff': 3072,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'huge': {
        'num_layers': 24,
        'd_model': 1024,
        'num_heads': 16,
        'dff': 4096,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},}
