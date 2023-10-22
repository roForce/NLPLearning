import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

#如果要在gpu上运行的话，需要同时将模型的参互和input数据转换成cuda支持的类型
#定义数据
x = torch.rand([50,1])
y = x * 3 + 0.8

#2.定义模型

class Lr(nn.Module):
    #这里注意，mac电脑的打出的下划线要短一些，实际的要长一些
#def __int__(self):
#def __init__(self):
    def __init__(self):

        super(Lr,self).__init__()
        #这里指全连接层的输入和输出，nn.Linear（输入的特征数量，输出的特征数量）是一个实现简单的线性模型的
        self.linear = nn.Linear(1,1)
        #完成一次向前计算的过程，需要定义forward方法到底是怎么算的
    def forward(self,x):
        out = self.linear(x)
        return out
#实例化模型,loss,和优化器
model = Lr()
#均方误差
criterion = nn.MSELoss()
#优化器类都是由torch.optim提供的，例如
#torch.optim.SGD,torch.optim.Adam(参数，学习率)，参数可以用model.parameters()来获取，获取模型中所有的requires_grad = True,也就是需要被追踪的参数
optimizer = optim.SGD(model.parameters(),lr=0.001)

#训练模型
for i  in range(30000):
    out = model(x) #获取预测值
    loss = criterion(y,out) # 计算损失
    optimizer.zero_grad()  #梯度归零
    loss.backward() #计算梯度
    optimizer.step() #更新梯度
    if(i + 1) % 20 == 0:
        print("Epoch[{}/{}],loss:{:.6f}".format(i,30000,loss.data))
#设置模型为评估模式，也就是预测模式
model.eval()
predict = model(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(),y.data.numpy(),c="r")
plt.plot(x.data.numpy(),predict)
plt.show()

