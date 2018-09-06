# -*- coding: utf-8 -*-
"""
CONFIG
------
对配置的封装
"""
from configparser import ConfigParser


class MyParser(ConfigParser):
    def __init__(self, defaults=None):
        super().__init__(defaults=defaults)

    # 这里重写了optionxform方法，直接返回选项名
    def optionxform(self, optionstr):
        return optionstr


_config = None


def get_config(config_file_path):
    """
    单例配置获取
    """
    global _config
    if not _config:
        config = MyParser()
        config.read(config_file_path, encoding='utf-8')
    else:
        config = _config
    return config
