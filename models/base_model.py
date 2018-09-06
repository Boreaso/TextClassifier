import abc

from utils.misc_utils import make_dirs


class BaseModel:

    def __init__(self,
                 model_path,
                 dictionary,
                 num_classes=2,
                 load_pretrained=False):
        """
        模型基类
        :param model_path: 模型保存路径
        :param dictionary: Dictionary字典类，用于分词
        :param load_pretrained: 是否加载预训练的模型
        """
        self.model_path = model_path
        self.dictionary = dictionary
        self.num_classes = num_classes
        self.load_pretrained = load_pretrained
        self.trained = False

        # 确保路径目录存在
        make_dirs(self.model_path)


    @abc.abstractmethod
    def train(self, input_file):
        """
        训练模型
        :param input_file: 预处理后的文本
        """
        pass

    @abc.abstractmethod
    def predict(self, doc):
        """
        预测文本
        :param doc: 输入文本
        :return: dict, key:label, value:probability
        """
        pass

    @abc.abstractmethod
    def test(self, input_file):
        """
        测试
        :param input_file: 测试文件
        :return: dict, key:评估指标, value: 指标值
        """
        pass
