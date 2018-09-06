import os
import time

import jieba
import numpy as np
from keras import Model
from keras.layers import Concatenate, Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.optimizers import Adam

from models.base_model import BaseModel
from utils.vocabulary import Vocabulary


class TextCNNModel(BaseModel):
    """TextCNN模型"""

    def __init__(self,
                 model_path,
                 dictionary,
                 vocab,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 batch_size=64,
                 learning_rate=0.001,
                 dropout=0.5,
                 l2_reg_lambda=0.0,
                 pre_embedding=False,
                 embedding_model=None,
                 embedding_initializer='uniform',
                 kernel_initializer='he_uniform',
                 load_pretrained=False,
                 validation_split=0.2,
                 epoch=10):
        """
        :param model_path: 模型保存路径
        :param dictionary: 字典类Dictionary对象
        :param vocab: Vocabulary对象
        :param sequence_length: 最大句子长度
        :param num_classes: 类别数
        :param vocab_size: 词汇表大小
        :param embedding_size: 词向量维数
        :param filter_sizes: 不同尺度的卷积核大小
        :param num_filters: 卷积核数量
        :param learning_rate: 学习率
        :param dropout: dropout比例
        :param l2_reg_lambda: l2正则化系数
        :param pre_embedding: 是否使用预训练的词向量
        :param load_pretrained: 是否加载预训练模型
        """
        super().__init__(model_path, dictionary, load_pretrained)
        self.vocab = vocab
        self.pre_embedding = pre_embedding
        self.embedding_model = embedding_model
        self.embedding_initializer = embedding_initializer
        self.kernel_initializer = kernel_initializer
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout = dropout
        self.validation_split = validation_split
        self.epoch = epoch

        # Build TextCNN Model.
        self.model = self._build_model()

        # Optimizer.
        self.optimizer = Adam(lr=self.learning_rate)

        # Compile.
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        if self.load_pretrained:
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

    def _build_model(self):
        """
        :rtype: Model
        """
        if self.pre_embedding:
            # Input layer.
            x_input = Input(shape=(self.sequence_length, self.embedding_size),
                            dtype='float32')
            embedding = x_input
        else:
            # Input layer.
            x_input = Input(shape=(self.sequence_length,), dtype='int32')

            # Embedding layer.
            # input: [None, sequence_length]
            # output: [None, sequence_length, embedding_size]
            embedding = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.embedding_initializer,
                input_length=self.sequence_length)(x_input)

        pool_layers = []

        for size in self.filter_sizes:
            # Convolution layers.
            # input: [None, sequence_length, embedding_size]
            # output: [None, sequence_length - size + 1, num_filters]
            conv = Conv1D(filters=self.num_filters,
                          kernel_size=size,
                          padding='valid',
                          strides=1,
                          kernel_initializer=self.kernel_initializer,
                          activation='relu')(embedding)
            # Pooling layers.
            # input: [None, sequence_length - size + 1, num_filters]
            # output: [None, 1, num_filters]
            pooling = MaxPooling1D(
                pool_size=self.sequence_length - size + 1)(conv)
            pool_layers.append(pooling)

        # Merge all convolution outputs through channels.
        # input: [len(filter_sizes), None, num_filters]
        # output: [None, 1, num_filters * len(filter_sizes)]
        merged = Concatenate(axis=-1)(pool_layers)

        # Flatten
        # input: [None, 1, num_filters * len(filter_sizes)]
        # output: [None, num_filters * len(filter_sizes)]
        flatten = Flatten()(merged)

        # Dropout layer.
        drop = Dropout(rate=self.dropout)(flatten)

        # Dense layer.
        # input: [None, num_filters * len(filter_sizes)]
        # output: [None, num_classes]
        x_output = Dense(
            self.num_classes,
            kernel_initializer=self.kernel_initializer,
            activation='softmax')(drop)

        return Model(inputs=x_input, outputs=x_output)

    def train(self, train_file):
        """
        Train textCNN with input data.
        :param input_file: input file path
        format: label word1 word2 word3 ...
        :return:
        """
        print('`%s` training...' % self.__class__.__name__)

        test_start = time.time()

        dataset = np.load(train_file)
        input_x, input_y = dataset['x'], dataset['y']

        self.model.fit(x=input_x, y=input_y,
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       validation_split=self.validation_split)

        print('`%s` train finished, time %s' %
              (self.__class__.__name__, time.time() - test_start))

        # 保存模型
        self.model.save(self.model_path)
        self.trained = True

    def predict(self, line):
        # 加载预训练模型
        if not self.trained:
            if os.path.exists(self.model_path):
                self.model.load_weights(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        seg = jieba.cut(line)

        # feature
        if self.pre_embedding:
            feature = [self.embedding_model.get_word_vector(w) for w in seg if w in self.vocab.vocab_table]
            if len(feature) < self.sequence_length:
                feature = feature + [self.embedding_model.get_word_vector(self.vocab.UNK)] * (
                            self.sequence_length - len(feature))
        else:
            feature = [self.vocab.vocab_table[w] for w in seg if w in self.vocab.vocab_table]
            if len(feature) < self.sequence_length:
                feature = feature + [0] * (self.sequence_length - len(feature))

        res = self.model.predict(np.array([feature]))
        res = {l: p for l, p in enumerate(res[0])}

        return res

    def test(self, test_file):
        print('`%s` testing...' % self.__class__.__name__)

        test_start = time.time()

        # 加载预训练模型
        if not self.trained:
            if os.path.exists(self.model_path):
                self.model.load_weights(self.model_path)
                self.trained = True
            else:
                raise FileNotFoundError('Model file `%s` not found.' % self.model_path)

        dataset = np.load(test_file)
        input_x, input_y = dataset['x'], dataset['y']

        res = self.model.evaluate(x=input_x, y=input_y, batch_size=self.batch_size)

        print('`%s` test finished, time %ss\n' %
              (self.__class__.__name__, time.time() - test_start))

        return {'acc': res[0]}


if __name__ == '__main__':
    config_file = '../configs/textcnn_config.conf'
    corpus_file = '../data/textcnn/corpus.txt'
    dataset_file = '../data/textcnn/dataset.npz'
    vocab_file = '../data/textcnn/validated_vocabs.txt'

    vocab = Vocabulary(vocab_file)

    text_cnn = TextCNNModel(
        model_path='../outputs/model/textcnn/model.ckpt',
        dictionary=None,
        vocab=vocab,
        sequence_length=40,
        num_classes=2,
        vocab_size=vocab.vocab_size,
        embedding_size=200,
        filter_sizes=[2, 3, 5],
        num_filters=32)

    text_cnn.model.summary()

    text_cnn.train(dataset_file)
