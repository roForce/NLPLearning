# from torchvision import transforms
# import numpy as np
#
# #测试图片数据的转换
# data = np.random.randint(0,255,size=12)
# img = data.reshape(2,2,3)
#
# print(img.shape)
# #转换成Tensor
# img_tensor = transforms.ToTensor()(img)
# #其中可以使用permute实现这一功能
# #实现的方法如下：
# '''
# img_tensor = torch.tensor(img)
# img_tensor = torch.permute(2,0,1)
# '''
# print(img)
# print(img_tensor)
# print(img_tensor.shape)



# #MINST是手写数据的识别，第一次使用的时候需要将Download参数设置为True，之后设置为False
# minist = MNIST(root="./data",train=True,download=False)
# ret = ToTensor()(minist[0][0])
# #可以看出将通道数，高，宽交换过来了
# print(ret.size())
# #2.构建模型
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
#数据的获得，数据集的处理
def get_dataloader(train=True,batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))  # 进行正则化 mean和std的形状和通道数相同
    ])
    dataset = MNIST(root='./data', train=train, transform=transform_fn)
    # 导入datatloader
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


class MinstModel(nn.Module):
    def __init__(self):
        super(MinstModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28,28)
        self.fc2 = nn.Linear(28,10)
        #这里还是得需要一些输入啊
    def forward(self,input):
        '''
        input:[bactchsize,1,28,28]
        :return:
        '''
        #修改形状
        x = input.view([-1,1*28*28])
        #全连接的操作，会调用__call_方法
        x = self.fc1(x)
        #激活函数进行激活,经过激活函数的处理，他的形状并不会发生响应的变化
        x  = F.relu(x)
        #输出层
        out = self.fc2(x)
        return F.log_softmax(out)

model = MinstModel()
optimizer = optim.Adam(model.parameters(),0.001)
#实现训练的过程，还需要实例化，我们的优化器类
if os.path.exists("./model/model.pk1"):
    model.load_state_dict(torch.load("./model/model.pk1"))
if os.path.exists("./model/model.pk2"):
    optimizer.load_state_dict(torch.load("./model/model.pk2"))

#训练模型
def train(epoch):
    #获取训练的数据集
    data_loader = get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad() #将参数置为零
        output = model(input) #调用模型得到预测值
        loss = F.nll_loss(output,target) #得到损失,带权损失
        loss.backward()  #反向传播
        optimizer.step() #更新梯度
        if idx % 10 == 0:
            print(epoch,idx,loss.item())
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pk1")
            torch.save(optimizer.state_dict(), "./model/model.pk2")

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))

if __name__ == "__main__":
    test()








