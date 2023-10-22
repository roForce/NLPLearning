'''
定义模型
'''
import torch.nn as nn
from lib import ws
class MyMode(nn.Module):
    def __init__(self):
        super(MyMode, self).__init__()
        nn.Embedding(len(ws),)