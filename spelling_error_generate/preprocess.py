import numpy as np
from numpy.random import choice
import codecs
from add_noise import add_noise_sentence
import json
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


print('load data')
with open('large_data/data_refine.txt','r',encoding="utf-8") as inp:
    sentences =  []
    for line in inp:
        if (line):
            # sentences.append(basic_tokenizer(json.loads(line)['sentence']))
            sentence = normalize('NFKC',line[:-1])
            sentence = basic_tokenizer(sentence)
            sentences.append(sentence)
print(len(sentences))
with open('out_data.txt','w') as f:
    for s in sentences:
        f.write(' '.join(s)+'\n')