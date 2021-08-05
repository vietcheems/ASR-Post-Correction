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
# from alphabet import VN_CHAR,near_char,EN_CHAR, noise_telex, noise_vni,closely_pronunciation,saigon_final2,saigon_final3, \
#                     like_pronunciation2, AEIOUYD_VN,dau_cau


def isword(word):
    for c in word:
        if c not in VN_CHAR:
            return 0
    return 1

pattern = re.compile(r'\d+\/\d+|\d+\.\d+|\w+\.+\w+|\w+\-+\w+|\w+|[{}]'.format(re.escape(punctuation)),re.UNICODE)
def basic_tokenizer(line):
    return pattern.findall(line)

def get_string(str, last=True, n=50):
    if not last:
        return str[:min(n,len(str))]
    else:
        return str[-min(n,len(str)):]

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

total = file_len('data_generate/transcript.txt')

with open('data_generate/transcript.txt','r',encoding="utf-8") as inp:
    count = 0
    line = inp.readline()
    chunk = 1
    sentences = []
    while line:
        if count >= 1000000*(chunk-1) and count < 1000000*(chunk):
            sentence = normalize('NFKC',line[:-1])
            sentence = basic_tokenizer(sentence)
            sentences.append(sentence)
            count +=1
        else:
            random.shuffle(sentences)
            for j in range(10):
                print(sentences[j])
            print(str(chunk), " ", str(len(sentences)))
            valid_sent = sentences[:int(total*0.1)]
            test_sent = sentences[int(total*0.1):int(total*0.2)]
            train_sent = sentences[int(total*0.2):]
            data ={}
            data['train'] = train_sent
            data['test'] = test_sent
            data['valid'] = valid_sent
            with open('data_generate/data1.pickle','ab') as f:
                pickle.dump(data,f)
            sentences = []
            chunk += 1
        line = inp.readline()





    # for line in inp:
    #     for i in range(1,12):
    #         sentences =  []
    #         if (line) and count > 1000000*(i-1) and count < 1000000*i:
    #             # sentences.append(basic_tokenizer(json.loads(line)['sentence']))
    #             sentence = normalize('NFKC',line[:-1])
    #             sentence = basic_tokenizer(sentence)
    #             sentences.append(sentence)
    #             count +=1
    #         random.shuffle(sentences)
    #         for j in range(10):
    #             print(sentences[j])
    #         print(str(i), " ", str(len(sentences)))
    #         valid_sent = sentences[:100000]
    #         test_sent = sentences[100000:200000]
    #         train_sent = sentences[200000:]
    #         data ={}
    #         data['train'] = train_sent
    #         data['test'] = test_sent
    #         data['valid'] = valid_sent
    #         with open('new_full_11M.pickle','ab') as f:
    #             pickle.dump(data,f)
            

# random.shuffle(sentences)
# for i in range(10):
#     print(sentences[i])
# print(len(sentences))
# valid_sent = sentences[:800000]
# test_sent = sentences[800000:900000]
# train_sent = sentences[900000:]
# data ={}
# data['train'] = train_sent
# data['test'] = test_sent
# data['valid'] = valid_sent
# with open('new_full_11M.pickle','wb') as f:
#     pickle.dump(data,f)
