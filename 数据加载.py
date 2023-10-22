import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_path = r"./SMSSpamCollection"


class MyDataSet(Dataset):
    def __init__(self):
        self.lines = open(data_path).readlines()
    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        # 分开标签
        cur_line = self.lines[index].strip()
        label = cur_line[:4].strip()
        print(label)
        content = cur_line[4:].strip()
        # fan
        return label, content

    def __len__(self):
        # 返回数据的总数量
        return len(self.lines)


if __name__ == '__main__':
    my_dataset = MyDataSet()
    # shuffle为True的时候表示是打乱的
    data_loader = DataLoader(dataset=my_dataset, batch_size=2,shuffle=True)
    for i in data_loader:
        print(i)
        break

    #enumerate可以遍历参数索引
    for index,(label,context) in enumerate(data_loader):
        print(index,label,context)