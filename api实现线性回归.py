import  torch
import torch.nn as nn
from torch.optim import SGD
#准备数据
x = torch.rand([500,1])
y_true = 3 * x  + 0.8
#定义模型
class MyLinear(nn.Module):
    def __init__(self):
        # 继承父类的init
        super(MyLinear,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        out = self.linear(x)
        return out


#实例化模型
my_linear = MyLinear()
#计算loss的函数
loss_fn = nn.MSELoss()

optimizer = SGD(my_linear.parameters(),lr=0.001)

#循环，进行梯度下降，参数的更新
for i in range(2000):
    y_predict = my_linear(x)

    loss = loss_fn(y_predict,y_true)
    #梯度置为0
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #参数的更新,相当于更新w,和b的值
    optimizer.step()
    if i % 50 == 0:
        print(loss.item(),list(my_linear.parameters()))