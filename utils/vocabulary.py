import os

__all__ = ["load_vocab",
           "Vocabulary"]


def load_vocab(vocab_file):
    """加载词汇文件"""
    vocab = []
    with open(vocab_file, "r", encoding='utf-8') as file:
        vocab_size = 0
        for _word in file:
            vocab_size += 1
            vocab.append(_word.strip())
    return vocab, vocab_size


class Vocabulary:
    UNK = "<unk>"
    UNK_ID = 0

    def __init__(self, vocab_file):
        # vocab_file, vocab_size
        self.vocab_file, self.vocab_size = self._validate_vocab_file(vocab_file)

        # generate tables
        self.vocab_table = self._get_vocab_table()

    def _validate_vocab_file(self, vocab_file):
        """合法化词汇文件，文件开头三个词汇必须依次为 <unk>, <s>, </s>"""

        print("# Validating file '%s' " % vocab_file)

        if os.path.exists(vocab_file):
            print("  Vocab file %s exists" % vocab_file)
            vocab, vocab_size = load_vocab(vocab_file)

            assert len(vocab) >= 1
            if vocab[0] != self.UNK:
                print("  The first vocab words %s is not %s" % (vocab[0], self.UNK))
                vocab = [self.UNK] + vocab
                vocab_size += 1
                validated_vocab_file = os.path.join(
                    os.path.dirname(vocab_file),
                    "validated_" + os.path.basename(vocab_file))

                with open(validated_vocab_file, "w", encoding='utf-8') as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = validated_vocab_file
        else:
            raise ValueError("  vocab_file '%s' does not exist." % vocab_file)

        vocab_size = len(vocab)
        return vocab_file, vocab_size

    def _get_vocab_table(self):
        # 根据文件生成词汇-index映射
        assert self.vocab_file

        if os.path.exists(self.vocab_file):
            vocab_file = open(self.vocab_file, encoding='utf-8')
            vocab_table = {}
            for i, vocab in enumerate(vocab_file):
                vocab_table[vocab.strip('\n')] = i
        else:
            raise ValueError("vocab_file '%s' does not exists" % self.vocab_file)

        return vocab_table
