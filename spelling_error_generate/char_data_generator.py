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
def data_gen(sents,context_len):
    data = []
    pre = 0
    #lỗi mất space
    for idx,sent in enumerate(sents):
        for i in range(1,len(sent)):
            if isword(sent[i]) and isword(sent[i-1]):
                before = get_string(' '.join(sent[:i-1]),True,context_len)
                after = get_string(' '.join(sent[i+1:]),False,context_len)
                inp =[]
                out = []
                out.append(sent[i-1]+' '+sent[i])
                inp.append(before+'$'+sent[i-1]+sent[i]+'&'+after)
                x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
                data.append(x)

    print(len(data)-pre)
    # pre=len(data)
    # lỗi đổi chỗ kí tự
    # for idx,sent in enumerate(sents):
    #     for i in range(0,len(sent)):
    #         if isword(sent[i]) and len(sent[i])>3:
    #             for j in range(0, len(sent[i])):
    #                 before = sent[i-1] if i!=0 else ""
    #                 after = sent[i+1] if i!=len(sent)-1 else ""
    #                 inp =[]
    #                 out = []
    #                 inp.append(before+'$'+sent[i][:j]+sent[i][j+1:]+'&'+after)
    #                 out.append(sent[i])
    #                 x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
    #                 data.append(x)
    # print(len(data)-pre)
    # pre=len(data)
    # #tu viet dung
    for idx,sent in enumerate(sents):
        for i in range(0,len(sent)):
            if isword(sent[i]) and len(sent[i])>1:
                before = get_string(' '.join(sent[:i]),True,context_len)
                after = get_string(' '.join(sent[i+1:]),False,context_len)
                inp =[]
                out = []
                inp.append(before+'$'+sent[i]+'&'+after)
                out.append(sent[i])
                x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
                data.append(x)
    string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
    print(len(data)-pre)
    pre=len(data)
    # lỗi với telex
    for idx,sent in enumerate(sents):
        for i in range(0,len(sent)):
            if isword(sent[i]) and len(sent[i])>1:
                word = sent[i]
                letter = [str(j) for j in word if j in string_list]
                for letter_i in letter:
                    tmp1 = noise_telex(letter_i)
                    r = random.random()
                    if r < 0.9:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'str':
                                word = word.replace(letter_i, tmp1[0])
                            else:
                                word = word.replace(
                                    letter_i, tmp1[0][0])
                        else:
                            if type(tmp1[1]).__name__ == 'str':
                                word = word.replace(letter_i, tmp1[1])
                            else:
                                word = word.replace(
                                    letter_i, tmp1[1][0]) + tmp1[1][1]
                    else:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'list':
                                word = word.replace(
                                    letter_i, tmp1[0][1])
                        else:
                            word = word.replace(
                                letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
                before = get_string(' '.join(sent[:i]),True,context_len)
                after = get_string(' '.join(sent[i+1:]),False,context_len)
                inp =[]
                out = []
                inp.append(before+'$'+word+'&'+after)
                out.append(sent[i])
                x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
                if random.random()>0.5:
                    data.append(x)
    print(len(data)-pre)
    pre=len(data)
    # lỗi với vni
    '''
    for idx,sent in enumerate(sents):
        for i in range(0,len(sent)):
            if isword(sent[i]) and len(sent[i])>1:
                word = sent[i]
                letter = [str(j) for j in word if j in string_list]
                for letter_i in letter:
                    tmp1 = noise_vni(letter_i)
                    r = random.random()
                    if r < 0.9:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'str':
                                word = word.replace(letter_i, tmp1[0])
                            else:
                                word = word.replace(
                                    letter_i, tmp1[0][0])
                        else:
                            if type(tmp1[1]).__name__ == 'str':
                                word = word.replace(letter_i, tmp1[1])
                            else:
                                word = word.replace(
                                    letter_i, tmp1[1][0]) + tmp1[1][1]
                    else:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'list':
                                word = word.replace(
                                    letter_i, tmp1[0][1])
                        else:
                            word = word.replace(
                                letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
                before = get_string(' '.join(sent[:i]),True,context_len)
                after = get_string(' '.join(sent[i+1:]),False,context_len)
                inp =[]
                out = []
                inp.append(before+'$'+word+'&'+after)
                out.append(sent[i])
                x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
                if random.random()>0.5:
                    data.append(x)
    '''
    # print(len(data)-pre)
    # pre=len(data)
    # loi dau cau
    # for idx,sent in enumerate(sents):
    #     for i in range(0,len(sent)):
    #         if isword(sent[i]) and len(sent[i])>2:
    #             before = get_string(' '.join(sent[:i]),True,context_len)
    #             after = get_string(' '.join(sent[i+1:]),False,context_len)
    #             word = sent[i]
    #             tmp = [j for j,char in enumerate(word) if char in AEIOUYD_VN]
    #             if len(tmp) > 0:
    #                 c = tmp[0]
    #                 dau = [dau_cau[i] for i in dau_cau if word[c] in dau_cau[i]]
    #                 # doi_dau = choice(dau[0],3,replace= False)
    #                 doi_dau = dau[0][0]
    #                 for doi_dau in choice(dau[0],2,replace= False):
    #                     wword = word[:c] + doi_dau + word[c+1:]                
    #                     inp =[]
    #                     out = []
    #                     inp.append(before+'$'+wword+'&'+after)
    #                     out.append(sent[i])
    #                     x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
    #                     data.append(x)

    # Các lỗi liên quan đến phát âm n,l,ch,tr

    # char1 = ['l', 'n', 'x', 's', 'r', 'd', 'v']
    # char2 = ["ch", 'tr', 'gi']
        
    # for idx,sent in enumerate(sents):
    #     for i in range(0,len(sent)):
    #         if isword(sent[i]) and len(sent[i])>1 and sent[i][:2] in char2 or sent[i][:1] in char1:
    #             before = get_string(' '.join(sent[:i]),True,context_len)
    #             after = get_string(' '.join(sent[i+1:]),False,context_len)

    #             word = sent[i]
    #             words = []
    #             if word[:2] in char2:
    #                 for s in closely_pronunciation[word[:2]]:
    #                   words.append(s+word[2:])
    #             elif word[:1] in char1 and word[:2] !='ng' and word[:2] !='nh':
    #                 for s in closely_pronunciation[word[0]]:
    #                     words.append(s+ word[1:])
    #             if unidecode.unidecode(word[:2]) in ['ri','di','vi']:
    #                 words = [word.replace('gi','g') for word in words]
    #             for w in words:
    #                 inp =[]
    #                 out = []
    #                 inp.append(before+'$'+w+'&'+after)
    #                 out.append(sent[i])
    #                 x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
    #                 data.append(x)


    # print(len(data)-pre)
    # pre=len(data)
    # lỗi đổi chỗ với kí tự gần đó trên bàn phím
    # for idx,sent in enumerate(sents):
    #     for i in range(0,len(sent)):
    #         if isword(sent[i]) and len(sent[i])>3:
    #             before = sent[i-1] if i!=0 else ""
    #             after = sent[i+1] if i!=len(sent)-1 else ""
    #             for j in np.random.choice(range(len(sent[i])),3,replace=False):
    #                 if sent[i][j] in EN_CHAR:
    #                     inp =[]
    #                     out = []
    #                     inp.append(before+'$'+sent[i][:j] + choice(near_char[sent[i][j]]) +sent[i][j+1:]+'&'+after)
    #                     out.append(sent[i])
    #                     x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
    #                     data.append(x)
    # print(len(data)-pre)
    # pre=len(data)
    # lỗi lặp lại kí tự
    # for idx,sent in enumerate(sents):
    #     for i in range(0,len(sent)):
    #         if isword(sent[i]) and len(sent[i])>3:
    #             before = get_string(' '.join(sent[:i]),True,context_len)
    #             after = get_string(' '.join(sent[i+1:]),False,context_len)
    #             for j in np.random.choice(range(len(sent[i])),2,replace=False):
    #                 if sent[i][j] in EN_CHAR and j>1:
    #                     inp =[]
    #                     out = []
    #                     inp.append(before+'$'+sent[i][:j] +np.random.randint(1,4)*sent[i][j]+sent[i][j:]+'&'+after)
    #                     out.append(sent[i])
    #                     x = {'tid':idx,'index' : idx, 'input': inp, 'output': out}
    #                     data.append(x)
    random.shuffle(data)
    # print(len(data)-pre)
    # pre=len(data)
    data = [dt for dt in data if len(dt['output'][0])<30 ]
    print(len(data))
    for i in range(len(data)):
        data[i]['index'] = i
        data[i]['tid'] = i
    return data

# with open('data_110_splited.json','r',encoding='utf8') as test:
#     test = test.read()
#     data =json.loads(test)

# train_sent = data['train']
# test_sent = data['test']
# valid_sent = data['valid']

print('load data')
# with open('large_data/data_refine.txt','r',encoding="utf-8") as inp:
#     sentences =  []
#     for line in inp:
#         if (line):
#             # sentences.append(basic_tokenizer(json.loads(line)['sentence']))
#             sentence = normalize('NFKC',line[:-1])
#             sentence = basic_tokenizer(sentence)
            
#             sentences.append(sentence)

# random.shuffle(sentences)
# for i in range(10):
#     print(sentences[i])
# print(len(sentences))
# valid_sent = sentences[:100000]
# test_sent = sentences[100000:200000]
# train_sent = sentences[200000:]
# data ={}
# data['train'] = train_sent
# data['test'] = test_sent
# data['valid'] = valid_sent
# with open('large_data/data.pickle','wb') as f:
#     pickle.dump(data,f)


with open('11Mdata/data.pickle','rb') as f:
    data = pickle.load(f)
train_sent = data['train'][:150000]
test_sent  = data['test'][:30000]
valid_sent = data['valid'][:30000]
with open('large_data/train_char150K.json','w',encoding='utf8') as train:
    print('train'+str(len(train_sent)))
    train_data = data_gen(train_sent,15)
    print(train_data[0:10])
    json.dump(train_data,train,ensure_ascii=False)
with open('large_data/valid_char.json','w',encoding='utf8') as valid:
    print('valid')
    valid_data = data_gen(valid_sent,15)
    print(valid_data[0:10],len(valid_data))
    json.dump(valid_data,valid,ensure_ascii=False)
with open('large_data/test_char.json','w',encoding='utf8') as test:
    print('test')
    test_data = data_gen(test_sent,15)
    print(test_data[0:10],len(test_data))
    json.dump(test_data,test,ensure_ascii=False)

print('done')
