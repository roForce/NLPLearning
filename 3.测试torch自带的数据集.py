from torchvision.datasets import MNIST
#MINST是手写数据的识别，第一次使用的时候需要将Download参数设置为True，之后设置为False
minist = MNIST(root="./data",train=True,download=False)
print(minist[0][0].show())