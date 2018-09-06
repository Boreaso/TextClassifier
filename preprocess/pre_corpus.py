import os


def combine_files(dir_path, dest_path):
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))
    with open(dest_path, mode='w', encoding='utf-8') as dest_file:
        files = os.listdir(dir_path)
        for file in files:
            path = os.path.join(dir_path, file)
            text = open(path, mode='r', errors='ignore').read()
            text = text[8:] if 'content' in text else text
            text = text.replace('\n', '').replace('\r', '')
            dest_file.writelines(text + '\n')


def transform(pos_path, neg_path, dest_path):
    """
    把文本转换为label + sentence的形式
    :param pos_path: 正类文本文件路径
    :param neg_path: 负类文本文件路径
    :param dest_path: 目标文件路径
    """
    pos_file = open(pos_path, encoding='utf-8')
    neg_file = open(neg_path, encoding='utf-8')
    dest_file = open(dest_path, mode='w', encoding='utf-8')

    for line in pos_file:
        line = '1 ' + line.replace(' ', '')
        dest_file.writelines(line)

    for line in neg_file:
        line = '0 ' + line.replace(' ', '')
        dest_file.writelines(line)

    neg_file.close()
    pos_file.close()
    dest_file.close()


if __name__ == '__main__':

    parent = '../data/corpus'
    dirs = os.listdir(parent)

    for base_dir in dirs:
        base_dir = os.path.join(parent, base_dir)
        combine_files('%s/pos' % base_dir, '%s/pos.txt' % base_dir)
        combine_files('%s/neg' % base_dir, '%s/neg.txt' % base_dir)

    for base_dir in dirs:
        pos_path = os.path.join(parent, base_dir, 'pos.txt')
        neg_path = os.path.join(parent, base_dir, 'neg.txt')
        dest_name = os.path.basename(base_dir) + '.txt'
        dest_path = os.path.join(parent, base_dir, dest_name)
        transform(pos_path, neg_path, dest_path)
