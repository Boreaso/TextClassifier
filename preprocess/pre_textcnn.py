import os

import fastText as ft
import jieba
import numpy as np
from keras.utils import to_categorical

from utils.dictionary import Dictionary
from utils.misc_utils import make_dirs
from utils.vocabulary import Vocabulary


def get_embedding_model(file, model_path, **kwargs):
    # 加载/训练词向量模型
    if os.path.exists(model_path):
        model = ft.load_model(model_path)
    else:
        model = ft.train_unsupervised(file, **kwargs)
        model.save_model(model_path)
    return model


def corpus_to_id_dataset(input_file,
                         dataset_file,
                         vocab: Vocabulary,
                         sequence_length,
                         num_classes):
    if os.path.basename(dataset_file).split('.')[-1] != 'npz':
        raise ValueError('Invalid extension name.')

    with open(input_file, encoding='utf-8') as input_f:
        input_x, input_y = [], []

        for line in input_f:
            line = line.split()
            # feature
            seg = list(jieba.cut(sentence=line[1]))[:sequence_length]
            feature = [vocab.vocab_table[w] for w in seg if w in vocab.vocab_table]
            if len(feature) < sequence_length:
                feature = feature + [0] * (sequence_length - len(feature))
            input_x.append(feature)
            # label
            label = to_categorical(int(line[0]), num_classes)
            input_y.append(label)

        input_x, input_y = np.array(input_x), np.array(input_y)
        np.savez(dataset_file, x=input_x, y=input_y)


def corpus_to_embedding_dataset(input_file,
                                dataset_file,
                                embedding_model,
                                vocab: Vocabulary,
                                sequence_length,
                                num_classes):
    if os.path.basename(dataset_file).split('.')[-1] != 'npz':
        raise ValueError('Invalid extension name.')

    with open(input_file, encoding='utf-8') as input_f:
        input_x, input_y = [], []

        for line in input_f:
            line = line.split()
            # feature
            seg = list(jieba.cut(sentence=line[1]))[:sequence_length]
            feature = [embedding_model.get_word_vector(w) for w in seg if w in vocab.vocab_table]
            if len(feature) < sequence_length:
                feature = feature + [embedding_model.get_word_vector(vocab.UNK)] * (sequence_length - len(feature))
            input_x.append(feature)
            # label
            label = to_categorical(int(line[0]), num_classes)
            input_y.append(label)

        input_x, input_y = np.array(input_x), np.array(input_y)
        np.savez(dataset_file, x=input_x, y=input_y)


def _make_id_dataset(input_file,
                     train_file,
                     test_file,
                     vocab: Vocabulary,
                     sequence_length,
                     num_classes,
                     test_split=0.2):
    """
    构建训练集和测试集。
    :param input_file: 语料数据，格式为：第一个字符为label，其余为文字数据
    :param train_file: 训练集文件路径
    :param test_file: 测试集文件路径
    :param test_split: 测试集比例
    :return:
    """
    print('`TextCNNModel` making dataset...')

    if not os.path.exists(input_file):
        raise FileNotFoundError('File `%s` not found.' % input_file)

    lines = []
    with open(input_file, encoding='utf-8') as file:
        for line in file:
            lines.append(line)

    # 生成数据集
    np.random.seed(123)
    train_set, test_set = [], []
    if test_split > 0:
        test_set = set(np.random.choice(
            lines, int(len(lines) * test_split), replace=False))
    train_set = set(lines) - test_set

    # 保存数据集
    train_tmp_file = os.path.join(
        os.path.dirname(train_file),
        str(os.path.basename(train_file).split('.')[0]) + '_tmp.txt')
    test_tmp_file = os.path.join(
        os.path.dirname(test_file),
        str(os.path.basename(test_file).split('.')[0]) + '_tmp.txt')
    if len(train_set) > 0:
        # 保存中间文件
        with open(train_tmp_file, mode='w', encoding='utf-8') as file:
            for train_line in train_set:
                file.writelines(train_line)
        # 生成数据集
        corpus_to_id_dataset(train_tmp_file, train_file, vocab, sequence_length, num_classes)

    if len(test_set) > 0:
        # 保存中间文件
        with open(test_tmp_file, mode='w', encoding='utf-8') as file:
            for test_line in test_set:
                file.writelines(test_line)
        # 生成数据集
        corpus_to_id_dataset(test_tmp_file, test_file, vocab, sequence_length, num_classes)


def _make_embedding_dataset(input_file,
                            input_seg_file,
                            train_file,
                            test_file,
                            embedding_model_path,
                            vocab: Vocabulary,
                            sequence_length,
                            num_classes,
                            test_split=0.2,
                            **embedding_kwargs):
    """
    构建训练集和测试集。
    :param input_file: 语料数据，格式为：第一个字符为label，其余为文字数据
    :param train_file: 训练集文件路径
    :param test_file: 测试集文件路径
    :param test_split: 测试集比例
    :return:
    """
    print('`TextCNNModel` making dataset...')

    if not os.path.exists(input_file):
        raise FileNotFoundError('File `%s` not found.' % input_file)

    lines = []
    with open(input_file, encoding='utf-8') as file:
        for line in file:
            lines.append(line)

    # 生成数据集
    np.random.seed(123)
    train_set, test_set = [], []
    if test_split > 0:
        test_set = set(np.random.choice(
            lines, int(len(lines) * test_split), replace=False))
    train_set = set(lines) - test_set

    # 保存数据集
    train_tmp_file = os.path.join(
        os.path.dirname(train_file),
        str(os.path.basename(train_file).split('.')[0]) + '_tmp.txt')
    test_tmp_file = os.path.join(
        os.path.dirname(test_file),
        str(os.path.basename(test_file).split('.')[0]) + '_tmp.txt')

    # 获得预训练embedding模型
    embedding_model = get_embedding_model(
        input_seg_file, embedding_model_path, **embedding_kwargs)

    if len(train_set) > 0:
        # 保存中间文件
        with open(train_tmp_file, mode='w', encoding='utf-8') as file:
            for train_line in train_set:
                file.writelines(train_line)
        # 生成数据集
        corpus_to_embedding_dataset(
            train_tmp_file, train_file, embedding_model,
            vocab, sequence_length, num_classes)

    if len(test_set) > 0:
        # 保存中间文件
        with open(test_tmp_file, mode='w', encoding='utf-8') as file:
            for test_line in test_set:
                file.writelines(test_line)
        # 生成数据集
        corpus_to_embedding_dataset(
            test_tmp_file, test_file, embedding_model,
            vocab, sequence_length, num_classes)


def make_dataset(input_file,
                 train_file,
                 test_file,
                 vocab: Vocabulary,
                 sequence_length,
                 num_classes,
                 test_split=0.2,
                 word_embedding=False,
                 input_seg_file=None,
                 embedding_model_path=None,
                 **embedding_kwargs):
    """
    生成数据集。
    :param input_file: 原始未处理语料数据，每行首字符为label，其余为语料数据
    :param train_file: 训练数据集路径
    :param test_file: 测试数据集路径
    :param vocab: Vocabulary对象
    :param sequence_length: 固定语句长度，长句截断处理，短句用0填充
    :param num_classes: 类别数量
    :param test_split: 测试集比例
    :param word_embedding: 是否预训练词嵌入向量
    :param input_seg_file: 已经进行分词的语料数据
    :param embedding_model_path: 词嵌入模型路径
    :param embedding_kwargs: 词嵌入模型参数
    """
    # 确保保存路径目录存在
    make_dirs(train_file)
    make_dirs(test_file)
    make_dirs(embedding_model_path)

    if word_embedding:
        _make_embedding_dataset(
            input_file=input_file,
            input_seg_file=input_seg_file,
            train_file=train_file,
            test_file=test_file,
            embedding_model_path=embedding_model_path,
            vocab=vocab,
            sequence_length=sequence_length,
            num_classes=num_classes,
            test_split=test_split,
            **embedding_kwargs)
    else:
        _make_id_dataset(
            input_file=input_file,
            train_file=train_file,
            test_file=test_file,
            vocab=vocab,
            sequence_length=sequence_length,
            num_classes=num_classes,
            test_split=test_split)


if __name__ == '__main__':
    config_file = '../configs/textcnn_config.conf'
    corpus_file = '../data/textcnn/corpus.txt'
    train_file = '../data/textcnn/dataset'
    test_file = '../data/textcnn/test_dataset'
    vocab_file = '../data/textcnn/validated_vocabs.txt'

    dictionary = Dictionary(config_file)
    # dictionary.cut()

    vocab = Vocabulary(vocab_file)

    make_dataset(input_file=corpus_file,
                 train_file=train_file,
                 test_file=test_file,
                 vocab=vocab,
                 sequence_length=40,
                 num_classes=2)
