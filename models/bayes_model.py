import os
import sys
import time

from snownlp import sentiment

from models.base_model import BaseModel


class BayesModel(BaseModel):

    def __init__(self,
                 dictionary,
                 model_path,
                 load_pretrained=False,
                 num_classes=2,
                 epoch=10):
        """
        模型参数
        """
        super().__init__(model_path, dictionary,
                         num_classes=num_classes,
                         load_pretrained=load_pretrained)

        if self.num_classes > 2:
            raise ValueError('`BayesModel` only support binary classification.')

        self.epoch = epoch
        self.model = sentiment.Sentiment()

        if self.load_pretrained:
            model_path = self.model_path
            if sys.version_info[0] == 3 and not self.model_path.endswith('.3'):
                model_path = self.model_path + '.3'
            if os.path.exists(model_path):
                self.model.load(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

    def train(self, input_file):
        """
        训练。
        :param input_file: 训练数据集路径
        """
        print('`%s` training...' % self.__class__.__name__)

        train_start = time.time()

        with open(input_file, encoding='utf-8') as file:
            neg_docs, pos_docs = [], []
            # 构建样本
            for line in file:
                line = line.split()
                label, doc = line[0], line[1]
                if label == '0':
                    neg_docs.append(doc)
                else:
                    pos_docs.append(doc)
        # 训练
        for _ in range(self.epoch):
            self.model.train(
                neg_docs=neg_docs, pos_docs=pos_docs)

        self.model.save(self.model_path)
        self.trained = True

        print('`%s` train finished, time %ss' %
              (self.__class__.__name__, time.time() - train_start))

    def predict(self, doc):
        """
        预测文本分类
        :param doc: 单条文本内容
        :return: dict, key:类别, value:概率
        """
        if not self.model:
            raise EnvironmentError('Model not trained.')

        # 加载预训练模型
        if not self.trained:
            model_path = self.model_path
            if sys.version_info[0] == 3 and not self.model_path.endswith('.3'):
                model_path = self.model_path + '.3'
            if os.path.exists(model_path):
                self.model.load(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        seg_doc = self.dictionary.cut(sentence=doc)
        res = self.model.classify(seg_doc)

        return {0: 1 - res,
                1: res}

    def test(self, input_file):
        """
        输出待分类文档分类。
        :param input_file: 测试文件
        :return:
            测试正确率，acc
            正类测试正确率：pos_acc
            负类测试正确率：neg_acc
        """
        if not self.trained:
            model_path = self.model_path
            if sys.version_info[0] == 3 and not self.model_path.endswith('.3'):
                model_path = self.model_path + '.3'
            if os.path.exists(model_path):
                self.model.load(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        print('`%s` testing...' % self.__class__.__name__)
        test_start = time.time()

        neg_docs, pos_docs = [], []
        with open(input_file, encoding='utf-8') as file:
            # 构建样本
            for line in file:
                line = line.split()
                label, doc = line[0], line[1]
                if label == '0':
                    neg_docs.append(doc)
                else:
                    pos_docs.append(doc)

        neg_total, pos_total = len(neg_docs), len(pos_docs)
        neg_correct, pos_correct = 0, 0

        for doc in neg_docs:
            if self.model.classify(doc) < 0.5:
                neg_correct += 1
        for doc in pos_docs:
            if self.model.classify(doc) > 0.5:
                pos_correct += 1

        neg_acc = neg_correct / neg_total
        pos_acc = pos_correct / pos_total
        acc = (neg_correct + pos_correct) / (neg_total + pos_total)

        print('`%s` test finished, time %ss' %
              (self.__class__.__name__, time.time() - test_start))

        return {'acc': acc,
                'pos_acc': pos_acc,
                'neg_acc': neg_acc}
