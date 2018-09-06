import os

import numpy as np

from utils.misc_utils import make_dirs


def make_dataset(input_file,
                 train_file,
                 test_file,
                 test_split=0.2):
    """
    构建训练集和测试集。
    :param input_file: 语料数据，格式为：第一个字符为label，其余为文字数据
    :param train_file: 训练集文件路径
    :param test_file: 测试集文件路径
    :param test_split: 测试集比例
    :return:
    """
    print('`FastTextModel` making dataset...')

    # 确保保存路径目录存在
    make_dirs(train_file)
    make_dirs(test_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError('File `%s` not found.' % input_file)

    lines = []
    with open(input_file, encoding='utf-8') as file:
        for line in file:
            lines.append(line)

    # 构造数据集
    np.random.seed(123)
    train_set, test_set = [], []
    if test_split > 0:
        test_set = set(np.random.choice(
            lines, int(len(lines) * test_split), replace=False))
    train_set = set(lines) - test_set

    # 保存数据集
    with open(train_file, mode='w', encoding='utf-8') as file:
        for train_line in train_set:
            file.writelines(train_line)
    with open(test_file, mode='w', encoding='utf-8') as file:
        for test_line in test_set:
            file.writelines(test_line)


