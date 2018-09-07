import os
import time

import fastText as ft

from models.base_model import BaseModel


class FastTextModel(BaseModel):

    def __init__(self,
                 model_path,
                 dictionary,
                 load_pretrained=False,
                 num_classes=2,
                 **kwargs):
        """
        模型参数

        * input_file             training file path (required)
        * output                 output file path (required)
        * label_prefix           label prefix ['__label__']
        * lr                     learning rate [0.1]
        * lr_update_rate         change the rate of updates for the learning rate [100]
        * dim                    size of word vectors [100]
        * ws                     size of the context window [5]
        * epoch                  number of epoch [5]
        * min_count              minimal number of word occurences [1]
        * neg                    number of negatives sampled [5]
        * word_ngrams            max length of word ngram [1]
        * loss                   loss function {ns, hs, softmax} [softmax]
        * bucket                 number of buckets [0]
        * minn                   min length of char ngram [0]
        * maxn                   max length of char ngram [0]
        * thread                 number of threads [12]
        * t                      sampling threshold [0.0001]
        * silent                 disable the log output from the C++ extension [1]
        * encoding               specify input_file encoding [utf-8]
        * pretrained_vectors     pretrained word vectors (.vec file) for supervised learning []
        """
        super().__init__(model_path, dictionary, load_pretrained=load_pretrained)

        self.num_classes = num_classes
        self.kwargs = kwargs
        self.model = None

        if load_pretrained:
            if os.path.exists(model_path):
                self.model = ft.load_model(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

    def train(self, train_file):
        """
        训练模型。
        :param train_file: 输入数据集文件路径
            format: label word1 word2 word3 ...
        """
        print('%s training...' % self.__class__.__name__)
        train_start = time.time()
        self.model = ft.train_supervised(train_file, **self.kwargs)
        self.model.save_model(self.model_path)
        self.trained = True
        print('`%s` train finished, time %ss' %
              (self.__class__.__name__, time.time() - train_start))

    def predict(self, doc):
        """
        预测输入文本。
        :param doc: 输入文本数据。
        :return: dict，key: 类别标记， value: 文本输入key类的概率
        """
        # 加载预训练模型
        if not self.trained:
            if os.path.exists(self.model_path):
                self.model = ft.load_model(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        seg_doc = self.dictionary.cut(sentence=doc)
        res = self.model.predict(seg_doc, k=self.num_classes)

        res = {int(l[-1]): p for l, p in zip(res[0], res[1])}
        res = sorted(res.items(), key=lambda x: x[0])
        res = {k: v for k, v in res}

        return res

    def test(self, input_file):
        # 加载预训练模型
        if not self.trained:
            if os.path.exists(self.model_path):
                self.model = ft.load_model(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        test_start = time.time()
        print('`%s` testing...' % self.__class__.__name__)

        res = self.model.test(input_file)

        print('`%s` test finished, time %ss' %
              (self.__class__.__name__, time.time() - test_start))

        return {'number': res[0],
                'precision': res[1],
                'recall': res[2]}
