import os
import re
import math
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

        q_words = {
            'q1_person': ['谁', '那个', '哪个'],
            'q1_time': ['那年', '时间', '哪年', '何时', '多久', '时候', '年'],
            'q1_amount': ['多', '几', '多少', '第几'],
            'q1_place': ['哪儿', '哪家', '哪里人', '哪里', '那家', '那里人', '那里'],
            'q1_result': ['怎么', '为什么', '为何', '如何', '何'],
            'q1_judge': ['是否', '还是', '吗'],
            'q0_other': ['哪些', '那些', '干什么'],
            'q0_definition': ['什么样', '什么', '怎么样', '怎样'],
        }
        self.question_words = []
        self.word2key = {}

        for k, v in q_words.items():
            self.question_words += v
            for _v in v:
                self.word2key[_v] = k

        self.stop_words = set()
        with open('../data/all-stop-word.txt') as f_stop:
            for i in f_stop.readlines():
                self.stop_words.add(i.strip())
        self.articles = []

    def cut_sentence(self, sent, stop=False):
        """
        句子分词
        :param sent: 
        :param stop: 
        :return: 
        """
        if stop:
            words = list(filter(lambda x: x not in self.stop_words, list(self.segmentor.segment(sent.strip()))))
        else:
            words = list(self.segmentor.segment(sent.strip()))
        return words

    def get_question_type(self, question):
        """
        获取问题类型
        :param question: 
        :return: 
        """
        q_type = ''
        words = self.cut_sentence(question)
        flag = False
        for w in self.question_words:
            if w in words:
                flag = True
                q_type = self.word2key[w]
                break
        if not flag:
            # print(i, words)
            q_type = 'other'
        print(q_type)

    def word_count(self, sentences):
        """
        篇章中的词频统计
        :param sentences: 句子列表
        :return: 
        """
        all_words = []
        for i in sentences:
            all_words += self.cut_sentence(i, True)
        word_count = {}
        for i in all_words:
            if i in word_count:
                word_count[i] += 1
            else:
                word_count[i] = 1
        return word_count, sum(word_count.values())

    def read_train_set(self, file_path):
        """
        读取测试文件
        :param file_path: 文件路径
        :return: 
        """
        with open(file_path) as f_in:
            last_q = ''
            article = {
                'question': '',
                'result': '',
                'sentences': []
            }
            for i in f_in.readlines():
                line = i.strip().split('\t')
                if last_q == line[1]:
                    article['sentences'].append(line[2])
                    if int(line[0]) == 1:
                        article['result'] = line[2]
                else:
                    self.articles.append(article)
                    article = {'question': line[1], 'result': '', 'sentences': []}
                last_q = line[1]
            self.articles.append(article)
        self.articles = self.articles[1:]
        print(len(self.articles))
        print(self.articles[0])

    def tf_idf(self):
        with open('../data/question_word.txt') as f_in:
            pass

    def analysis_question(self, index, debug=True):
        if len(self.articles) <= 0:
            return
        article = self.articles[index]
        q_words = self.cut_sentence(article['question'], True)
        true_result = ''.join(self.cut_sentence(article['result'], True))
        if debug:
            print('q', self.cut_sentence(article['question'], True))
            print('q', article['question'])

            print('a', self.cut_sentence(article['result'], True))
            print('a', true_result)
        # print(q_words)
        # 候选答案句切词
        l_words = [self.cut_sentence(line, True) for line in article['sentences']]
        # 计算关键词idf
        idf = {}
        for word in q_words:
            count = 0
            for line in l_words:
                if word in line:
                    count += 1
            idf[word] = count
        idf = {k: math.log(len(l_words) * 1.0/(v + 1)) if len(l_words) > 0 else 0 for k, v in idf.items()}
        # print(idf)

        line2score = {}
        for line in l_words:
            score = 0
            for word in q_words:
                # 计算关键词tf
                tf = 0
                delta = 1
                if len(re.findall('\d+', word)) > 0:
                    delta = 3
                for i in line:
                    if i == word:
                        tf += 1
                if len(line) == 0:
                    tf = 0
                else:
                    tf = (tf * 1.0 * delta) / len(line)
                score += tf * idf[word]
            line2score[''.join(line)] = score
        res = sorted(line2score.items(), key=lambda x: x[1], reverse=True)
        if debug:
            for i in res:
                print(i[1], i[0])
        if len(res) > 0:
            for i in range(len(res)):
                if res[i][0] == true_result:
                    return i + 1
            return 0
        else:
            return 0

def test(sentence):
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

    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    netags = recognizer.recognize(words, postags)
    arcs = parser.parse(words, postags)  # 句法分析

    res = zip(words, postags, netags, arcs)
    for i in res:
        print(','.join(i[:3]), str(i[3].head) + ':' + i[3].relation)
    # print(list(words))
    # print(list(postags))
    # print(list(netags))
    # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))

    # tree = s_parser.parse(words)
    # for i in tree:
    #     print(i)
    #
    # dependency_parser = stanford.StanfordDependencyParser(model_path=stanford_model_path)
    # res = list(dependency_parser.parse(words))
    # for row in res[0].triples():
    #     print(row)


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
        'q1_person': ['谁', '那个', '哪个'],
        'q1_time': ['那年', '时间', '哪年', '何时', '多久', '时候', '年'],
        'q1_amount': ['多', '几', '多少', '第几'],
        'q1_place': ['哪儿', '哪家', '哪里人', '哪里', '那家', '那里人', '那里'],
        'q1_result': ['怎么', '为什么', '为何', '如何', '何'],
        'q1_judge': ['是否', '还是', '吗'],
        'q0_other': ['哪些', '那些', '干什么'],
        'q0_definition': ['什么样', '什么', '怎么样', '怎样'],
    }
    word2key = {}
    type2questions = {'other': []}

    question_words = []
    print(q_words.items())
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
        for w in question_words:
            if w in words:
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
    # test('2006年7月27日，360安全卫士正式推出。')
    # get_all_questions()
    my_parser = Parser()
    # analysis_questions(my_parser)
    # my_parser.get_question_type('缓刑适用于几年以下的有期徒刑')
    my_parser.read_train_set('../data/BoP2017-DBQA.train.txt')
    count = 0
    for i in range(len(my_parser.articles)):
        res = my_parser.analysis_question(i, debug=False)
    # for i in range(10):
    #     res = my_parser.analysis_question(i, debug=True)
        if res == 0:
            count += 0
        else:
            count += 1.0 / res
    print('score', count/len(my_parser.articles))

    # my_parser.analysis_question(0)
