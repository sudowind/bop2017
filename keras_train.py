# -*- coding: utf-8 -*-

import tensorflow
import keras
import codecs

from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Input, LSTM, Dense, Embedding, concatenate
from keras.models import Model

import numpy as np

# 路径
f1 = codecs.open("bop2017/src/cut_words_train.txt", "r", encoding="utf-8-sig")
f2 = codecs.open("bop2017/src/cut_words_dev.txt", "r", encoding="utf-8-sig")

# train question/answer/result
tq = []
ta = []
tr = []

# develop question/answer/result
dq = []
da = []
dr = []

# get data, fuck the encode
lines = f1.readlines()
for line in lines:
    tr.append(int(line.split('\t')[0]))
    tq.append(line.split('\t')[1].encode("utf-8"))
    ta.append(line.split('\t')[2].encode("utf-8"))

lines = f2.readlines()
for line in lines:
    dr.append(int(line.split('\t')[0]))
    dq.append(line.split('\t')[1].encode("utf-8"))
    da.append(line.split('\t')[2].encode("utf-8"))

# 记录数据集大小
train_num = len(tq)
dev_num = len(dq)

# 合并方便keras文本预处理
texts = tq + ta + dq + da
results = tr + dr


# maxn = 0
# count = 0
# for i in texts:
#     if maxn < len(i):
#         maxn = len(i)
#     if len(i.split()) > 300:
# #         print(i)
#         count+=1


    
maxn = 300
#print(count, len(texts), (count+0.0)/len(texts))




tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxn)
labels = to_categorical(np.asarray(results))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



# restore train / test data
tq1 = data[:train_num]
ta1 = data[train_num: train_num*2]
tr1 = labels[:train_num]

dq1 = data[train_num*2:train_num*2+dev_num]
da1 = data[train_num*2+dev_num:]
dr1 = labels[train_num:]

# print(len(tq1), len(ta1), len(tr1))
# print(len(dq1), len(da1), len(dr1))

# Found 343193 unique tokens.
# Shape of data tensor: (658868, 3096)
# Shape of label tensor: (329434, 2)


# model start

q_input = Input(shape=(300, ))
a_input = Input(shape=(300, ))

shared_embedding = Embedding(input_dim = 343193 + 1, output_dim=100, input_length=300, mask_zero=True)

embed_q = shared_embedding(q_input)
embed_a = shared_embedding(a_input)

lstm_q = LSTM(100)(embed_q)
lstm_a = LSTM(100)(embed_a)

merged_vector = concatenate([lstm_q, lstm_a], axis=-1)

is_answer = Dense(2, activation='sigmoid')(merged_vector)

model = Model(inputs=[q_input, a_input], outputs=is_answer)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([tq1, ta1], tr1, epochs=10, validation_data=([dq1, da1], dr1))
