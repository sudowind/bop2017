import os
from nltk.parse import stanford

from src.config import *


def test():
    os.environ['STANFORD_PARSER'] = STANFORD_PARSER_PATH
    os.environ['STANFORD_MODELS'] = STANFORD_MODELS_PATH
    os.environ['JAVAHOME'] = JAVA_HOME
    stanford_model_path = CHINESE_MODEL_PATH
    s_parser = stanford.StanfordParser(model_path=stanford_model_path)

    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    from pyltp import Parser
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型

    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

    from pyltp import Segmentor
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

    from pyltp import Postagger
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

    from pyltp import NamedEntityRecognizer
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型

    sentence = '国立清水高级中学科学馆的外观是什么样的'
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    netags = recognizer.recognize(words, postags)
    arcs = parser.parse(words, postags)  # 句法分析

    print(list(words))
    print(list(postags))
    print(list(netags))
    print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

    tree = s_parser.parse(words)
    for i in tree:
        print(i)

    dependency_parser = stanford.StanfordDependencyParser(model_path=stanford_model_path)
    res = list(dependency_parser.parse(words))
    for row in res[0].triples():
        print(row)


if __name__ == '__main__':
    test()
