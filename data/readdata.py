import numpy as np
from easydict import EasyDict
import torch
# 读取npy文件
motion = EasyDict(torch.load('data/motions/h1_walk.npy'))
motion2 = EasyDict(torch.load('data/motions/dog_fast_run_02_004_worldpos.npy'))

print(motion.keys())