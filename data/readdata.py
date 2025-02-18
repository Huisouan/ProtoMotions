import numpy as np
from easydict import EasyDict
import torch
# 读取npy文件
motion = EasyDict(torch.load('data/motions/h1_walk.npy'))


print(motion.keys())