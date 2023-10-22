#主要讲述Pytorch中的tensor的常用的方法
#获取tensor中的数据（当tensor中只有一个元素可以用的时候）
import torch
import numpy as np

#获取tensor中只有一个元素 tensor.item()
a = torch.tensor([[1,2,3],[4,5,6]])

#转换为numpy数组
#a.numpy()
#获取
#a.shape(1)
#获取某一个维度的数据
#a.size(1)#获取对应维度的数值

#形状的修改
#a.view((3,4)) #类似于numpy中的reshape，是一种浅拷贝，仅仅是形状发生改变

#
t3 = torch.tensor(np.arange(24).reshape(2,3,4))
print(t3.transpose(0,1))

t3 = t3.new_ones(5,3)
print(t3)