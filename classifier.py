import os
from argparse import ArgumentParser

import fastText as ft

from models.bayes_model import BayesModel
from models.fasttext_model import FastTextModel
from models.textcnn_model import TextCNNModel
from preprocess import pre_bayes, pre_fasttext, pre_textcnn
from utils.config import get_config
from utils.dictionary import Dictionary
from utils.vocabulary import Vocabulary


def convert(s):
    try:
        result = eval(s)  # for int, long, bool and float
    except Exception:
        result = s  # for str
    return result


def get_params(config, section='model'):
    model_params = {}
    for k, v in config.items(section, raw=True):
        model_params[k] = convert(v)
    return model_params


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default='',
                        help="Config file path.")
    parser.add_argument("--mode", type=str, default='train',
                        help="Running mode. train | test")

    flags, _ = parser.parse_known_args()

    config_path = flags.config_path

    # flags.mode = 'predict'
    # config_path = 'configs/bayes.conf'
    # config_path = 'configs/fasttext.conf'
    # config_path = 'configs/textcnn.conf'

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            'Config file `%s` not found, please specify param `config_path`.' % config_path)

    if not config_path:
        raise ValueError('Parameter `config_path` must be specified.')

    # ConfigParser对象
    config = get_config(config_path)

    # 模型参数
    model_params = get_params(config)

    # 其他相关参数
    model_type = config.get('global', 'model_type')
    model_path = config.get('data', 'model_path')
    corpus_file = config.get('data', 'corpus_path')
    seg_corpus_file = config.get('data', 'seg_corpus_path')
    vocab_file = config.get('data', 'vocabs_path')
    train_file = config.get('data', 'train_dataset_path')
    test_file = config.get('data', 'test_dataset_path')
    test_split = config.get('global', 'test_split')
    test_split = float(test_split)

    # 切词
    dictionary = Dictionary(config_path)
    if not os.path.exists(seg_corpus_file) or not os.path.exists(vocab_file):
        dictionary.cut()

    # 增加额外参数
    if model_type.lower() == 'textcnn':
        model_creator = TextCNNModel
        # vocab
        vocab_file = config.get('data', 'vocabs_path')
        vocab = Vocabulary(vocab_file)
        model_params['vocab'] = vocab
        model_params['vocab_size'] = vocab.vocab_size
        # 生成数据集
        sequence_length = model_params['sequence_length']
        num_classes = model_params['num_classes']
        embedding_model_path = config.get('data', 'embedding_model_path')
        embedding_params = get_params(config, section='word_embedding')
        pre_textcnn.make_dataset(
            corpus_file, train_file, test_file, vocab,
            sequence_length, num_classes, test_split,
            word_embedding=True,
            input_seg_file=seg_corpus_file,
            embedding_model_path=embedding_model_path,
            **embedding_params)
        # 词向量模型
        model_params['embedding_model'] = ft.load_model(embedding_model_path)
    elif model_type.lower() == 'fasttext':
        model_creator = FastTextModel
        corpus_file = config.get('data', 'seg_corpus_path')
        pre_fasttext.make_dataset(corpus_file, train_file, test_file, test_split)
    else:
        model_creator = BayesModel
        pre_bayes.make_dataset(corpus_file, train_file, test_file, test_split)

    model = model_creator(
        dictionary=dictionary,
        model_path=model_path,
        **model_params)

    if flags.mode == 'train':
        model.train(train_file)
    elif flags.mode == 'test':
        res = model.test(test_file)
        print('Result:\n', res)
    elif flags.mode == 'predict':
        while True:
            txt = input("Input: ")
            if txt == 'exit':
                break
            if txt == '':
                continue
            res = model.predict(txt)
            print('Result %s' % res)
    else:
        raise ValueError('Invalid mode `%s`' % flags.mode)
