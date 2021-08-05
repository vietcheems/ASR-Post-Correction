import numpy as np
from numpy.random import choice
import codecs
from add_noise import add_noise_sentence
import json
from sklearn.model_selection import train_test_split
import random
# from nltk.tokenize import word_tokenize
from alphabet import *
import re
from string import punctuation
import unidecode
import pickle
from unicodedata import normalize
import pandas as pd
pattern = re.compile(r'\d+\/\d+|\d+\.\d+|\w+\.+\w+|\w+\-+\w+|\w+|[{}]'.format(re.escape(punctuation)),re.UNICODE)
def basic_tokenizer(line):
    return pattern.findall(line)

df = pd.read_csv('/home3/phungqv/post_correction/data/100k/word2num/preprocessed.csv')

# with open('data_generate/transcript.txt','r',encoding="utf-8") as inp:
#     sentences =  []
#     for line in inp:
#         if (line):
#             # sentences.append(basic_tokenizer(json.loads(line)['sentence']))
#             sentence = normalize('NFKC',line[:-1])
#             sentence = basic_tokenizer(sentence)
            
#             sentences.append(sentence)

sentences =  []
for line in df.input:
    if (line):
        # sentences.append(basic_tokenizer(json.loads(line)['sentence']))
        try:
            sentence = normalize('NFKC',line[:])
            sentence = basic_tokenizer(sentence)
            sentences.append(sentence)
        except Exception as e:
            print(sentence)
            print(e.__class__)
        

# random.shuffle(sentences)
# for i in range(10):
#     print(sentences[i])
# print(len(sentences))
# valid_sent = sentences[:int(len(sentence)*0.1)]
# test_sent = sentences[int(len(sentence)*0.1):int(len(sentence)*0.2)]
# train_sent = sentences[int(len(sentence)*0.2):]
train_sent = sentences
valid_sent = []
test_sent = []
data ={}
data['train'] = train_sent
data['test'] = test_sent
data['valid'] = valid_sent
with open('/home3/phungqv/post_correction/data/100k/word2num/input.pickle','wb') as f:
    pickle.dump(data,f)