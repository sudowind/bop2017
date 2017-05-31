from pyltp import Segmentor
from pyltp import Postagger
from src.config_local import *
segmentor = Segmentor()  # 初始化实例
segmentor.load(LTP_DATA_DIR + '/cws.model')  # 加载模型
postagger = Postagger()
postagger.load(LTP_DATA_DIR + '/pos.model')

stop_word = []
count_question = 0
count_correct = 0
count_wrong = 0
correct_filename = '../data/correct_question.txt'
wrong_filename = '../data/wrong_question.txt'


def cut_words(sentence):
    global segmentor
    words = segmentor.segment(sentence)
    word_list = list(words)
    for word in word_list:
        if word in stop_word:
            word_list.remove(word)
    return word_list


def postag():
    pass


def count_same(questions, answers):
    count = 0
    for word in questions:
        if word in answers:
            count += 1
    return count


def cal_answer(question, answers, data):
    global count_question, count_correct, count_wrong, correct_filename, wrong_filename
    count_question += 1
    question_words = cut_words(question)
    size = len(answers)
    print(size)
    standard = 0
    flag = 0
    max = -1
    for k in range(0,size):
        answer = data[k]
        if answer == '1':
            standard = k
        answer_words = cut_words(answers[k])
        count = count_same(question_words, answer_words)
        if count > max:
            max = count
            flag = k
    if flag == standard:
        count_correct += 1
        with open(correct_filename, 'a+') as f:
            f.write(question + '\t' + answers[flag] + '\n')
    else :
        count_wrong += 1
        with open(wrong_filename, 'a+') as f:
            f.write(question + '\t' + answers[flag] + '\t' + answers[standard] + '\n')
    print(count_question)

if __name__ == '__main__':
    filename = '../data/BoP2017-DBQA.train.txt'
    base_question = ''
    base_answers = []
    base_data = []
    with open('../data/all-stop-word.txt', 'r') as f:
        while 1:
            line = f.readline().strip()
            if not line:
                break
            stop_word.append(line)

    with open(filename, 'r') as f:
        while 1:
            line = f.readline()
            if not line:
                break
            segments = line.split('\t')
            data = segments[0]
            question = segments[1]
            answer = segments[2]
            if question == base_question:
                base_answers.append(answer)
                base_data.append(data)
            elif base_question == '':
                base_question = question
                base_answers.append(answer)
                base_data.append(data)
            else:
                cal_answer(base_question, base_answers, base_data)
                base_question = question
                base_answers = []
                base_data = []
                base_answers.append(answer)
                base_data.append(data)


