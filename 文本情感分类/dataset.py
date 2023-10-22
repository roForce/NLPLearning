'''
完成数据集的准备
'''
from torch.utils.data import DataLoader,Dataset
import os
import re

def tokenlize(content):
    #删除所有的特殊的符号和标点符号
    re.sub("<.*?>"," ",content)
    filter = ['\t','\n','\x97','\x96','#','$','%','&','\.']
    content = re.sub("|".join(filter)," ",content)
    #strip()方法就是删除前后含有空格的部分
    tokens = [i.strip() for i in content.split()]
    return tokens
class ImdbDataSet(Dataset):
    def __init__(self,train = True):
        self.train_data_path = r"训练集路径"
        self.test_data_path = r"测试集路径"
        data_path = self.train_data_path if train else self.test_data_path
        #1.把所有的文件名放入列表
        temp_data_path = [os.path.join(data_path,"pos"),os.path.join(data_path,"neg")]
        #所有的评论文件的path
        self.total_file_path = []
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path,i) for i in file_name_list]
            self.total_file_path.extend(file_path_list)
    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        #获取label,选取倒数第二个文件名字也就是label标签
        label_str = file_path.split("\\")[-2]
        label = 0 if label_str == 'neg' else 1
        #获取内容,对句子再进行相应的分词的处理
        content = open(file_path).read()
        tokens = tokenlize(content)
        return tokens,label



    def __len__(self):
        return len(self.total_file_path)


def collate_fn(batch):
    """

    :param batch:[tokens,label],[token,label]
    :return:
    """
    content,label= list(zip(*batch))
    return content,label


def getdataLoader(train=True):
    imdb_dataset = ImdbDataSet()
    data_loader = DataLoader(imdb_dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    return data_loader
if __name__ == "__main__":
    pass


