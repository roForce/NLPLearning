import torch
import numpy as np
#直接使用torch进行转换
t1 = torch.tensor([1,2,3])
#使用nump转换
arr = np.arange(12).reshape(3,4)
arr = torch.tensor(arr)
#创建一个全为1的tensor
t2 = torch.ones([3,4])
#创建空数组
t2 = torch.empty([3,4])

#torch.zeros([3,4]),创建三行四列，全为0的tensor
t3 = torch.zeros([3,4])

#torch.rand([3,4]),创建一个三行四列的随机整数的tensor，随机值之间的区间是[0,1]
t3 = torch.rand([3,4])
#torch.randl(low = 0,high = 10,size = [3,4] ),创建以恶搞三行四列的随机整数的Tensor，随机值的区间
#为[0,10]
t4 = torch.randint(0,10,[3,4])

