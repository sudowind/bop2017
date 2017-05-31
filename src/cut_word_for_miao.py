# -*- coding: utf-8 -*-
from pyltp import Segmentor
#分词
def segment(segmentor, sentence):
    words = segmentor.segment(sentence)  # 分词
    words_list = list(words)
    return words_list



def read_file(file_name, segmentor):
	with open(file_name) as f:
		while 1:
			line = f.readline().strip()
			if not line:
				break
			segments = line.split('\t')
			question = segments[1]
			answer = segments[2]
			word_list = segment(segmentor, question)
			with open('cut_words_dev.txt', 'a+') as f1:
				for k in word_list:
					f1.write(k + ' ')
			word_list = segment(segmentor, answer)
			with open('cut_words_dev.txt', 'a+') as f1:
				for k in word_list:
					f1.write(k + ' ')
				f1.write('\n')

if __name__ == '__main__':
	segmentor = Segmentor()
	segmentor.load('ltp_data/cws.model')
	#filename = 'BoP2017-DBQA.train.txt'
	filename = 'BoP2017-DBQA.dev.txt'
	read_file(filename, segmentor)
