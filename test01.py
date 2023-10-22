#对一个y = 3 * x + 0.8 的数据进行简单的线性回归分析
import torch

from  matplotlib import pyplot as plt

learn_rate = 0.1
#1.准备数据
x = torch.rand([500,1])
y_true =  x * 3 + 0.8
#2。通过模型计算y_predict
#设置对应 的参数
w = torch.rand([1,1],requires_grad=True)
#造一个
b = torch.tensor(0,requires_grad=True,dtype=torch.float32)

#4.通过循环，反向传播，更新参数
for i in range(2000):
    # 3.计算loss
    # mean是算出均值
    y_predict = torch.matmul(x, w) + b
    #算出损失均值
    loss = (y_true - y_predict).pow(2).mean()
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()#反向传播
    w.data = w.data - learn_rate * w.grad
    b.data = b.data - learn_rate * b.grad
#    print("w,b,loss",w.item(),b.item(),loss)
plt.figure(figsize=(20,8))
plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
y_predict = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1),y_predict.detach().numpy().reshape(-1))
plt.show()