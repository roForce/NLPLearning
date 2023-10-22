"""
实现的是构建词典，实现吧句子转换成数字序列比昂将其翻转
"""
class Word2Sequence:
    UNK_TAG = "UNK"
    PAD_TAG =  "PAD"
    UNK = 0
    PAD = 0

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count = []

    def fit(self,sentence):
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1
    def build_vocab(self,min=5,max = None,max_feature=None):
        """
        生成词典
        :param min:最小出现的次数
        :param max: 最大的次数
        :param max_feature: 一共保留多少个词语
        :return:
        """
        #删除count中词频小雨min的词频
        if min is not None:
            self.count = {word: value for word,value in self.count if value > min}
        if max is not None :
            self.count = {word: value for word, value in self.count if value < max}
        #限制表流的词语数,取前maxfeature个
        if max_feature is not None:
            temp = sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_feature]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        #得到一个翻转的dict的字典
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentense,max_len = None):
        """
        把句子转换成序列
        :param sentense:
        :max_len :表示句子的最大长度
        :return:
        """
        if max_len is not None:
            if max_len > len(sentense):
                sentense = sentense  + [self.PAD_TAG]* (max_len - len(sentense))
            if max_len < len(sentense):
                sentense = sentense[:max_len]
        return [self.dict.get(word,self.UNK) for word in sentense]

    def inverse_transform(self,indices):
        """
        把序列转换成句子
        :param indices:
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]
    

