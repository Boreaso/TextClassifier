# -*- coding: utf-8 -*-
"""
DICT
-----
字典类封装
"""
import gzip
import os
from collections import Counter

import jieba
import numpy as np

from utils.config import get_config

_dictionary = None


class Dictionary:
    _jieba = None
    _vocabs = None

    def __init__(self, config_path):
        self.config_path = config_path

    @classmethod
    def _get_stop_words(cls):
        puncs = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                    u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")
        other = {'hellip', 'quot', 'ldquo', 'rdquo', 'mdash'}
        return puncs | other

    @classmethod
    def label_maps(cls, item):
        """
        标签转换
        """
        _maps = {'0': '__label__0',
                 '1': '__label__1',
                 '2': '__label__2',
                 '__label__0': 'negative',
                 '__label__1': 'positive',
                 '__label__2': 'neutral'}
        return _maps.get(item)

    def _load_user_dict(self):
        """
        加载用户词典
        """
        config = get_config(self.config_path)
        user_dict_path = config.get('data', 'user_dict_path')

        if user_dict_path:
            gr = gzip.open(user_dict_path)
            lines = gr.readlines()
            words = set([line.strip() for line in lines if line.strip()])
            user_dict = ['{} {} n'.format(word, len(word) * 1000) for word in words]
            buff_file = '\n'.join(user_dict)
            jieba.load_userdict(buff_file)
            gr.close()

        self._jieba = jieba

    def _get_vocabs(self, vocabs_path):
        """
        读取本地词典
        """
        if not self._vocabs:
            fr = open(vocabs_path, 'r', encoding='utf-8')
            line = fr.read()
            vocabs = line.split('\n')
            fr.close()
        else:
            vocabs = self._vocabs
        return vocabs

    def _cut_corpus(self, corpus_path, seg_corpus_path, sample, sample_corpus_path, vocabs_path):
        """
        语料分词
        """
        stop = self._get_stop_words()
        fr = open(corpus_path, 'r', encoding='utf-8')
        fw = open(seg_corpus_path, 'w', encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            line = line.split()
            words = [w.strip() for w in jieba.cut(line[1]) if w not in stop and w.strip()]
            fw.write('{} {}\n'.format(self.label_maps(line[0]), ' '.join(words)))
        fw.close()
        fr.close()

        if sample:
            self.sample(seg_corpus_path, sample_corpus_path, vocabs_path)
        else:
            fr = open(seg_corpus_path, 'r', encoding='utf-8')
            lines = fr.readlines()
            all_words = []
            for line in lines:
                all_words.extend(line.split())
            fw = open(vocabs_path, 'w', encoding='utf-8')
            fw.write('\n'.join(set(all_words)))
            fw.close()

    def _cut_sentence(self, sample, sentence, vocabs_path):
        """
        句子分词
        """
        words = [w.strip() for w in self._jieba.cut(sentence[1:]) if w.strip()]
        if sample and os.path.exists(vocabs_path):
            vocabs = self._get_vocabs(vocabs_path)
            words = [w for w in words if w in vocabs]
        else:
            stop = self._get_stop_words()
            words = [w for w in words if w not in stop]
        return '{} {}'.format(self.label_maps(sentence[0]), ' '.join(words))

    def cut(self, **kwargs):
        """
        语料分词
        """
        config = get_config(self.config_path)
        kwargs.setdefault('corpus_path', config.get('data', 'corpus_path'))
        kwargs.setdefault('seg_corpus_path', config.get('data', 'seg_corpus_path'))
        kwargs.setdefault('sample_corpus_path', config.get('data', 'sample_corpus_path'))
        kwargs.setdefault('vocabs_path', config.get('data', 'vocabs_path'))
        kwargs.setdefault('sample', config.getboolean('data', 'sample'))
        kwargs.setdefault('sentence', '')
        if not self._jieba:
            self._load_user_dict()
        if not kwargs.get('sentence'):
            kwargs.pop('sentence')
            self._cut_corpus(**kwargs)
        else:
            return self._cut_sentence(
                kwargs['sample'], kwargs['sentence'], kwargs['vocabs_path'])

    @classmethod
    def _min_freq_sample(cls, words, freq=5):
        """
        最小采样,去除词频过低的词
        """
        word_counts = Counter(words)
        # 剔除出现频率低的词, 减少噪音
        return [word for word in words if word_counts[word] > freq]

    @classmethod
    def _down_sample(cls, words, t=1e-5, threshold=0.75):
        """
        下采样,去除词频过高的词
        """
        # 统计单词出现频次
        word_counts = Counter(words)
        total_count = len(words)
        # 计算单词频率
        word_freqs = {w: c / float(total_count) for w, c in word_counts.items()}
        # 计算被删除的概率
        prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in word_counts}
        # 剔除出现频率太高的词
        train_words = [w for w in words if prob_drop[w] < threshold]
        return set(train_words)

    def sample(self, seg_corpus_path, sample_corpus_path, vocabs_path):
        """
        词采样
        """
        fr = open(seg_corpus_path, 'r', encoding='utf-8')
        lines = fr.readlines()
        all_words = []
        for line in lines:
            all_words.extend(line.split())
        temp_words = self._min_freq_sample(all_words)
        vocabs = self._down_sample(temp_words)

        fw = open(vocabs_path, 'w', encoding='utf-8')
        fw.write('\n'.join(vocabs))
        fw.close()

        fw = open(sample_corpus_path, 'w', encoding='utf-8')
        for line in lines:
            seg_words = line.split()
            sample_words = [w for w in seg_words[1:] if w in vocabs]
            if not sample_words:
                continue
            fw.write('{} {}\n'.format(seg_words[0], ' '.join(sample_words)))
        fw.close()

    def get_corpus_path(self, sample=None):
        """
        获取语料路径
        """
        config = get_config(self.config_path)
        if sample:
            corpus_path = config.get('data', 'sample_corpus_path')
        else:
            corpus_path = config.get('data', 'seg_corpus_path')
        return corpus_path
