import os
from nltk.parse import stanford

from src.config_local import *


class Parser:
    def __init__(self):
        os.environ['STANFORD_PARSER'] = STANFORD_PARSER_PATH
        os.environ['STANFORD_MODELS'] = STANFORD_MODELS_PATH
        os.environ['JAVAHOME'] = JAVA_HOME
        stanford_model_path = CHINESE_MODEL_PATH
        self.s_parser = stanford.StanfordParser(model_path=stanford_model_path)

        par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

        from pyltp import Parser
        self.parser = Parser()  # 初始化实例
        self.parser.load(par_model_path)  # 加载模型

        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

        from pyltp import Segmentor
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(cws_model_path)  # 加载模型

        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

        from pyltp import Postagger
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(pos_model_path)  # 加载模型

        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

        from pyltp import NamedEntityRecognizer
        self.recognizer = NamedEntityRecognizer()  # 初始化实例
        self.recognizer.load(ner_model_path)  # 加载模型


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


def get_all_questions():
    all_line = []
    with open('../data/BoP2017-DBQA.train.txt') as f_in:
        all_line = f_in.readlines()
    questions = set()
    for i in all_line:
        questions.add(i.strip().split('\t')[1])
    print(len(questions))
    with open('../data/train_questions.txt', 'w') as f_out:
        for i in questions:
            f_out.write(i + '\n')


def analysis_questions(parser):

    q_words = {
        'q_person': ['谁', '那个', '哪个'],
        'q_time': ['那年', '时间', '哪年', '何时', '多久', '时候', '年'],
        'q_amount': ['多', '几', '多少', '第几'],
        'q_place': ['哪儿', '哪家', '哪里人', '哪里', '那家', '那里人', '那里'],
        'q_result': ['怎么', '为什么', '为何', '如何', '何'],
        'q_judge': ['是否', '还是', '吗'],
        'q_other': ['哪些', '那些', '干什么'],
        'q_definition': ['什么样', '什么', '怎么样', '怎样'],
    }
    word2key = {}
    type2questions = {'other': []}

    question_words = []
    for k, v in q_words.items():
        question_words += v
        for _v in v:
            word2key[_v] = k
    print(question_words)
    print(word2key)

    all_line = []
    with open('../data/train_questions.txt') as f_in:
        all_line = f_in.readlines()
    no_q_words = []
    for i in all_line:
        words = list(parser.segmentor.segment(i.strip()))
        flag = False
        for w in words:
            if w in question_words:
                flag = True
                q_type = word2key[w]
                if q_type not in type2questions:
                    type2questions[q_type] = [(words, w, q_type)]
                else:
                    type2questions[q_type].append((words, w, q_type))
                break
        if not flag:
            # print(i, words)
            no_q_words.append((words, 'other'))
            type2questions['other'].append((words, 'other'))
    # print(len(no_q_words))
    for k, v in type2questions.items():
        print(k, len(v))
        for i in v[:10]:
            print(i)
    with open('../data/question_type.txt', 'w') as f_out:
        for k, v in type2questions.items():
            f_out.write(k + '\n')
            for i in v:
                tmp = ' '.join(i[0])
                f_out.write(tmp + '\t' + '\t'.join(i[1:]) + '\n')


if __name__ == '__main__':

    # get_all_questions()
    my_parser = Parser()
    analysis_questions(my_parser)
