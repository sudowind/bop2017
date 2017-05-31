# -*- coding: utf-8 -*-
from pyltp import Segmentor
#分词
def segmentor(sentence_set, word_dict):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load('ltp_data/cws.model')  # 加载模型
    for k in sentence_set:
    	words = segmentor.segment(k)  # 分词
    	words_list = list(words)
    	for word in words_list:
        	count_num = word_dict.get(word, 0)
        	word_dict[word] = count_num + 1

    segmentor.release()  # 释放模型
    return words_list

def read_file(file_name, questions, answers):
	with open(file_name) as f:
		while 1:
			line = f.readline().strip()
			if not line:
				break
			segments = line.split('\t')
			question = segments[1]
			answer = segments[2]
			questions.add(question)
			answers.add(answer)

#测试分词
if __name__ == '__main__':
	filename = 'BoP2017-DBQA.train.txt'
	questions = set()
	answers = set()
	read_file(filename, questions, answers)
	# with open('questions.txt', 'w') as f:
	# 	for k in questions:
	# 		f.write(str(k) + '\n')
	question_word = {}
	segmentor(questions, question_word)
	l = sorted(question_word.items(), key = lambda item:item[1],reverse = True)
	with open('question_word.txt', 'w') as f:
		for k in l:
			f.write(str(k[0]) + ' ' + str(k[1]) + '\n')
	# answer_word = {}
	# segmentor(answers, answer_word)
	