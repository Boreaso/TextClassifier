import os


def make_dirs(path):
    """保证指定路径的父文件夹存在"""
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
