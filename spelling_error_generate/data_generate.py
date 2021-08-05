# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import choice
import codecs
from add_noise import add_noise_sentence
import json
from sklearn.model_selection import train_test_split
import random
from nltk.tokenize import word_tokenize
from alphabet import VN_CHAR
import pickle
from string import punctuation
import os
NUMBER_KIND_OF_ERROR = 16
# RAITO_EACH_ERROR = [0.1,0.07,0.07,0.07,0.12,0.07,0.07,0.11,0.11,0.07,0.07,0.07,0]
#RAITO_EACH_ERROR = [0.1,0.07,0.07,0.1,0.1,0.07,0.08,0.1,0.08,0.07,0.08,0.08,0]
RAITO_EACH_ERROR = [0 for i in range(NUMBER_KIND_OF_ERROR)]

#Xóa kí tự
RAITO_EACH_ERROR[0] = 0.1
#Nhầm kí tự tương đồng
RAITO_EACH_ERROR[8] = 0.3
#thêm kí tự
RAITO_EACH_ERROR[13] = 0.1
#đổi dấu
RAITO_EACH_ERROR[14] = 0.4
#thay bộ phận câu
RAITO_EACH_ERROR[15]= 0.1

# RAITO_EACH_ERROR[8]= 0.5
# RAITO_EACH_ERROR[10]= 0.25
# RAITO_EACH_ERROR[10]= 0.25

# RAITO_EACH_ERROR[5]=  0.2
# RAITO_EACH_ERROR[6] = 0.2
# RAITO_EACH_ERROR[3] = 0.5

RANGE_RAITO_ERROR_EACH_SENTENTCE = [1,5]
NO_ERROR_SENTENCES_RATIO = 0.3
# NUMBER_KIND_OF_ERROR = 13
# RAITO_EACH_ERROR = [0.24,0.07,0.07,0,0.1,0.07,0.07,0.07,0.1,0.07,0.07,0.07,0]
# RANGE_RAITO_ERROR_EACH_SENTENTCE = [1,20]
# NO_ERROR_SENTENCES_RATIO = 0.2

def isword(word):
    """Check if every char in VN_char"""
    for c in word:
        if c.lower() not in VN_CHAR:
            return 0
    return 1
def split_sentences(sentences,max_length = 50):
    """Split setences into max_length-char chunks"""
    split_sents = []
    count=0
    for i,sentence in enumerate(sentences):
        # if i < 3:
        #     print(sentence)
        #     print(split_sents)
        #     print(len(split_sents))
        if len(sentence) <= 50:
            split_sents.append(sentence)
        else:
            split_sents.append(sentence[:50])
            count+=1
    # print(count)
    return split_sents

def find_error_location(sent_len):
    '''
    input: list length of sentences
    output: List of what kind of error in each sentences
    '''
    # Generate number of error in each sentence with random by random
    number_error_each_sent=[]
    for i in sent_len:
        s,e = RANGE_RAITO_ERROR_EACH_SENTENTCE
        # s = max(int(s*i/100),1) 
        # e = max(int(e*i/100),s+1)
        try:
            number_error_each_sent.append(choice(range(s,e)))
        except:
            print(s,e)
    # Make a error slot  
    total = np.sum(number_error_each_sent)
    print('total error {}'.format(total))
    slot = range(total)
    # Make a look up table to convert slot=>sent
    look_up = []
    for n_sent,i in enumerate(number_error_each_sent):
        for j in range(i):
            look_up.append(n_sent)
    number_error_each_kind = [int(total*ratio) for ratio in RAITO_EACH_ERROR]       
    print('số lỗi của mỗi loại dự kiến')
    print(number_error_each_kind)
    slot_error_each_kind = []
    # Random choose slot for each kind of error
    for e in number_error_each_kind:
        choiced_slot = choice(slot,e,replace = False)
        choiced_slot = np.sort(choiced_slot)
        choiced_slot = np.append(choiced_slot,total+10)
        cur = 0
        unchoiced = []
        for i in slot:
            if i < choiced_slot[cur]:
                unchoiced.append(i)
            else:
                cur+=1
        slot = unchoiced
        slot_error_each_kind.append(choiced_slot[:-1])
    # Convert slot to sentence
    kind_error_each_sentence = [[] for i in range(len(sent_len))]
    for i,error_slot in enumerate(slot_error_each_kind):
        for s in error_slot:
            kind_error_each_sentence[look_up[s]].append(i+1)
    return kind_error_each_sentence

def add_noise_list(sentences,output_file):
    """
    Input: sentences: list of sentences
    output_file: 
    Output: input_sentences: list of sentences with errors
            output_sentences: original (correct) sentences
            error_position_list: list of error position (labels for training)"""
    input_sentences = []
    output_sentences = []
    error_position_list = []
    number_error_each_kind = [0 for i in range(NUMBER_KIND_OF_ERROR)]
    
    # random.shuffle(sentences)
    no_error = int(len(sentences)*NO_ERROR_SENTENCES_RATIO)

    sent_len = [len(sent) for sent in sentences[:-no_error]] if no_error !=0 else [len(sent) for sent in sentences]
    list_error_per_sentence = find_error_location(sent_len)
    for i in range(no_error):
        list_error_per_sentence.append([])
    
    c = list(zip(sentences, list_error_per_sentence))

    # random.shuffle(c)

    sentences, list_error_per_sentence = zip(*c)
    for sentence, error_list in zip(sentences,list_error_per_sentence):
        error_position = [0 if isword(word) else -1 for i,word in enumerate(sentence)]
        input_sentence = sentence 
        output_sentence = sentence
        for e in error_list:
            input_sentence,output_sentence,error_position = add_noise_sentence(input_sentence,output_sentence,error_position,e)
        input_sentences.append(input_sentence)
        output_sentences.append(output_sentence)
        error_position_list.append(error_position)

    data = []
    with open(output_file,'w',encoding='utf8') as out:
        for i,(input_sent,output_sent,error) in enumerate(zip(input_sentences,output_sentences,error_position_list)):
            # error_sent = sent
            # for e in error:
            #     error_sent = add_noise(error_sent,e)
            x = {'tid':i,'index' : i, 'input': input_sent, 'output': output_sent,'error': error}
            # json.dump(x,out,ensure_ascii=False)
            # out.write('\n')
            data.append(x)
            for e in error:
                if (e >0):
                    number_error_each_kind[e-1]+=1
        json.dump(data,out,ensure_ascii=False)
    for dt in data[0:100:20]:
        print(dt['input'])
        print(dt['output'])    
        print(dt['error'])
    number_error_each_kind[2] //=2 
    print(number_error_each_kind)
    return input_sentences, output_sentences, error_position_list


def data_gen(input_file,train_file,valid_file,test_file):
    print('load data')
    with open(input_file,'rb') as inp:
        data = pickle.load(inp)
    print(len(data))
    
    #     sentences =  []
    #     for line in inp:
    #         sentences.append(json.loads(line)['sentence'])
    # sentences=sentences[0:150000:4]
    # duplicate sentences 3 times
    # sentences = sentences+sentences+sentences 
    # with open('data_110_splited.json','r',encoding='utf8') as test:
    #     test = test.read()
    #     data =json.loads(test)

    train_sentences = data['train'] #[:5000000]
    test_sentences = data['test']
    valid_sentences = data['valid']
    print(f'train length: {len(train_sentences)}')
    train_sentences =  split_sentences(train_sentences)
    valid_sentences = split_sentences(valid_sentences)
    test_sentences = split_sentences(test_sentences)   
    print(f'train length after splitting: {len(train_sentences)}')
    # print('max_len')
    # print(max([len(s) for s in train_sentences]))
    # print(max([len(s) for s in test_sentences]))
    # print(max([len(s) for s in valid_sentences]))
    # # test_sentences =[]
    # for s in sentences:
    #     words = basic_tokenizer(s)
    #     if 5<len(words)<120:
    #         test_sentences.append(words)
    # train_sentences,test_sentences = train_test_split(sentences,test_size=0.1)
    # duplicate sentences 3 times
    # train_sentences = train_sentences + train_sentences + train_sentences
    # test_sentences = test_sentences + test_sentences + test_sentences 
    # valid_sentences = valid_sentences + valid_sentences + valid_sentences


    print('train sentences {}'.format(len(train_sentences)))
    print('test sentences {}'.format(len(test_sentences)))
    print('valid_sentences {}'.format(len(valid_sentences)))
    print('find error position')
    
    print('add noise')
    _,_,_= add_noise_list(train_sentences,train_file)
    print('done creating train file')
    _,_,_= add_noise_list(valid_sentences,valid_file)
    _,_,_= add_noise_list(test_sentences,test_file)


import re
_WORD_SPLIT = re.compile("[“”…‘’.,{!?\"\'\][:};)(]")
stop_words = "\" \' [ ] (){.} , ! : ; ?".split(" ")
pattern = re.compile(r'\d+\/\d+|\d+\.\d+|\w+\.+\w+|\w+\-+\w+|\w+|[{}]'.format(re.escape(punctuation)),re.UNICODE)
def basic_tokenizer(line):
    return pattern.findall(line)
# def basic_tokenizer(sentence):
#     """Very basic tokenizer: split the sentence into a list of tokens."""
#     words = []
#     for space_separated_fragment in sentence.strip().split():
#         words.extend(_WORD_SPLIT.split(space_separated_fragment))
#     return [w for w in words if w not in stop_words and w != '' and w != ' ']


if __name__ == "__main__":
    dir = '/home3/phungqv/post_correction/data/100k/word2num/'
    data_gen(os.path.join(dir, 'input.pickle'), os.path.join(dir, 'train.json'), os.path.join('val.json'), os.path.join('test.json'))
    # print(basic_tokenizer('xin (chào) "tôi" là đức anhhbtt@gmaim.com ...'))
    # a= 'abc'
    # print('Done')
    # sentences = [['im','mồm']]
    # error = [[1]]
    # print(add_noise_list(sentences,error))
    