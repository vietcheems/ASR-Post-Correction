# -*- coding: utf-8 -*-

"""
    0. xoa 1 ki tu trong tu, khong tinh dau _
    1. doi cho 2 ki tu lien tiep trong 1 tu
    2. them _ trong 1 tu
    3. bot _ trong 1 tu , neu co nhieu thi chon ngau nhien
    4. gan nhau tren ban phim
    5. noise_telex
    6. noise_vni
    7. lap lai mot so nguyen am khong dung o dau am tiet

    8. cac am vi o dau cua am tiet gan giong nhau
    9. saigon phonology  am cuoi
    10. nga va hoi

    11. cac am vi o dau co cach phat am giong nhau trong mot so truong hop, vi du c q k
    12. sai vi tri dau va thay doi chu cai voi to hop cac dau neu am tiet co 1 nguyen am

    13. add 1 character to a word
    14. change the diacritic of a word
    15. remove a component of a word (thay phụ âm hoặc nguyên âm)
"""
import numpy as np 
from numpy.random import choice
import numpy.random as random
import unidecode
import copy
from alphabet import CONSONANT, DIACRITIC_LIST, NO_DIACRITIC_VN_CHAR, VN_CHAR, VOWEL,near_char,EN_CHAR, noise_telex, noise_vni,closely_pronunciation,saigon_final2,saigon_final3, \
                    like_pronunciation2, AEIOUYD_VN,bo_dau, HUYEN, SAC, HOI, NGA, NANG, AEIOUY_VN
MAX_LOOP = 50
                                                 

def add_noise_sentence(input_sentence,output_sentence,error_position,type_noise):
    input_sentence = copy.deepcopy(input_sentence)
    output_sentence = copy.deepcopy(output_sentence)
    if type_noise == 1:
        return delete_char(input_sentence,output_sentence,error_position)
    elif type_noise == 2:
        return swap_char(input_sentence,output_sentence,error_position)
    elif type_noise == 3:
        return add_space(input_sentence,output_sentence,error_position)
    elif type_noise == 4:
        return delete_space(input_sentence,output_sentence,error_position)
    elif type_noise == 5:
        return change_by_near(input_sentence,output_sentence,error_position)
    elif type_noise == 6:
        return add_telex_noise(input_sentence,output_sentence,error_position) #
    elif type_noise == 7:
        return add_vni_noise(input_sentence,output_sentence,error_position)  #
    elif type_noise == 8:
        return repeat_vowel(input_sentence,output_sentence,error_position)
    elif type_noise == 9:
        return spelling_mistake(input_sentence,output_sentence,error_position)
    elif type_noise == 10:
        return sai_gon_phonology(input_sentence,output_sentence,error_position) #
    elif type_noise == 11:
        return nga_or_hoi(input_sentence,output_sentence,error_position) #thay tất cả các dấu trong câu hay chỉ thay ngẫu nhiên 1 từ (thay 1 từ)
    elif type_noise == 12:
        return similar_wrong(input_sentence,output_sentence,error_position)
    elif type_noise == 13:
        return wrong_position_diacritic(input_sentence,output_sentence,error_position)
    elif type_noise == 14:
        return add_char(input_sentence, output_sentence, error_position)
    elif type_noise == 15:
        return change_diacritic(input_sentence, output_sentence, error_position)
    elif type_noise == 16:
        return replace_component(input_sentence, output_sentence, error_position)
    
def delete_char(input_sentence,output_sentence,error_position):
    """Error type 0"""
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>3]
    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        j = random.randint(0, len(word) - 1)
        word = word[:j] + word[j+1:]
        input_sentence[i] = word
        error_position[i] = 1

    return input_sentence,output_sentence,error_position

def swap_char(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>2]

    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        j = np.random.randint(1,len(word))
        word =  word[:j-1] + word[j:j+1] + word[j-1:j] + word[j+1:]
        input_sentence[i] = word
        error_position[i] = 2

    return input_sentence,output_sentence,error_position

def add_space(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>2]

    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        j = np.random.randint(1,len(word)-1)
        input_word = [word[:j],word[j:]] 
        output_word = [word,'']
        input_sentence = input_sentence[:i] + input_word +input_sentence[i+1:]
        output_sentence = output_sentence[:i] + output_word +output_sentence[i+1:]
        error_position = error_position[:i] + [3,3] + error_position[i+1:]

    
    return input_sentence,output_sentence,error_position

def delete_space(input_sentence,output_sentence,error_position):
    
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i in range(len(error_position)-1) if error_position[i]==0 and error_position[i+1]==0]

    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word1 = input_sentence[i]
        word2 = input_sentence[i+1]
        input_word = [word1+word2]
        output_word = [word1+' '+word2]
        input_sentence = input_sentence[:i] + input_word +input_sentence[i+2:]
        output_sentence = output_sentence[:i] + output_word +output_sentence[i+2:]
        error_position = error_position[:i] + [4] + error_position[i+2:]

    return input_sentence,output_sentence,error_position
    
def change_by_near(input_sentence,output_sentence,error_position):
    
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>1]
    
    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        x=0
        while(1):
            x+=1
            if x>MAX_LOOP:
                return input_sentence,output_sentence,error_position
            j = np.random.randint(0,len(word))
            if word[j] in EN_CHAR:
                break

        word = word[:j] + choice(near_char[word[j]]) +word[j+1:]
        input_sentence[i] = word
        error_position[i] = 5

    return input_sentence,output_sentence,error_position

def add_telex_noise(input_sentence,output_sentence,error_position):
    def check_fix(word):
        for c in word:
            if c in string_list:
                return True
        return False
    string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
    maybe_fix = [i for i,word in enumerate(output_sentence) if check_fix(word) and error_position[i]==0]
    
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
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
        input_sentence[i] = word
        error_position[i] = 6

    return input_sentence,output_sentence,error_position 

def add_vni_noise(input_sentence,output_sentence,error_position):
    def check_fix(word):
        for c in word:
            if c in string_list:
                return True
        return False
    string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
    maybe_fix = [i for i,word in enumerate(output_sentence) if check_fix(word) and error_position[i]==0]
    
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
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
        input_sentence[i] = word
        error_position[i] = 7

    return input_sentence,output_sentence,error_position 

def repeat_vowel(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>2]
    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        x=0
        while(1):
            x+=1
            if x>MAX_LOOP:
                return input_sentence,output_sentence,error_position
            j = np.random.randint(2,len(word))
            if word[j] in EN_CHAR:
                break

        word = word[:j] +np.random.randint(1,4)*word[j]+word[j:]
        input_sentence[i] = word
        error_position[i] = 8

    return input_sentence,output_sentence,error_position

def spelling_mistake(input_sentence,output_sentence,error_position):
    char1 = ['x', 's', 'r', 'd', 'm', 'n', 'l']
    char2 = ['ch', 'tr', 'gi']
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and (word[:2] in char2 or word[:1] in char1) and word[:2]!='ng' and word[:2] !='nh']
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        words = []
        if word[:2] in char2:
            for s in closely_pronunciation[word[:2]]:
                words.append(s+word[2:])
        elif word[:1] in char1 and word[:2] !='ng' and word[:2] !='nh':
            for s in closely_pronunciation[word[0]]:
                words.append(s+ word[1:])
        if unidecode.unidecode(word[:2]) in ['ri','di','vi']:
            words = [word.replace('gi','g') for word in words]
        if len(words)==0:
            print(word)
        else:
            input_sentence[i]=choice(words)
            error_position[i] = 9

    return input_sentence,output_sentence,error_position

def sai_gon_phonology(input_sentence,output_sentence,error_position):
    sai_gon3 = ['inh', 'ênh', 'iên', 'ươn', 'uôn', 'iêt', 'ươt', 'uôt']
    sai_gon2 = ['ăn', 'an', 'ân', 'ưn', 'ắt', 'ât', 'ưt', 'ôn', 'un',
                    'ât', 'ưt', 'ôn', 'un', 'ôt', 'ut']

    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and (word[-3:] in sai_gon3 or word[-2:] in sai_gon2)]
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        if word[-2:] in sai_gon2:
            word = word[:-2] + saigon_final2[word[-2:]]
        else:
            word = word[:-3] + saigon_final3[word[-3:]]
        input_sentence[i]=word
        error_position[i] = 10

    return input_sentence,output_sentence,error_position

def nga_or_hoi(input_sentence,output_sentence,error_position):
    
    def check_fix(word):
        for c in word:
            if c in AEIOUYD_VN:
                return True
        return False

    string_list = ['ã', 'ả',
                'ẫ', 'ẩ',
                'ẵ', 'ẳ',
                'ẻ', 'ẽ',
                'ể', 'ễ',
                'ĩ', 'ỉ',
                'ũ', 'ủ',
                'ữ', 'ử',
                'õ', 'ỏ',
                'ỗ', 'ổ', 'ỡ', 'ở']
    swap = {'ã': 'ả', 'ả': 'ã', 'ẫ': 'ẩ', 'ẩ': 'ẫ',
            'ẵ': 'ẳ', 'ẳ': 'ẵ', 'ẻ': 'ẽ', 'ẽ': 'ẻ', 'ễ': 'ể', 'ể': 'ễ',
            'ĩ': 'ỉ', 'ỉ': 'ĩ', 'ũ': 'ủ', 'ủ': 'ũ', 'ữ': 'ử', 'ử': 'ữ',
            'õ': 'ỏ', 'ỏ': 'õ', 'ỗ': 'ổ', 'ổ': 'ỗ', 'ỡ': 'ở', 'ở': 'ỡ'}
    maybe_fix = [i for i,word in enumerate(output_sentence) if check_fix(word) and error_position[i]==0 and len(word)>2 ]
    if len(maybe_fix)==0:
        return input_sentence,output_sentence,error_position
    else:
        i = choice(maybe_fix)
        word = input_sentence[i]
        tmp = [j for j,char in enumerate(word) if char in AEIOUYD_VN]
        if len(tmp) > 0:
            c = tmp[0]
            dau = [bo_dau[i] for i in bo_dau if word[c] in bo_dau[i]]
            if word[-1] in ['p','t','c'] or word[-2:] =='ch':
                while(1):
                    doi_dau = choice([dau[0][0],dau[0][2],dau[0][3],dau[0][4]])
                    if doi_dau!= word[c]:
                        break
            else:
                if word[c] == dau[0][3]:
                    doi_dau = dau[0][4]
                else:
                    doi_dau= word[c]
            # doi_dau = dau[0][0]
            
            word = word[:c] + doi_dau + word[c+1:]
        if word != input_sentence[i]:
            input_sentence[i] = word 
            error_position[i] = 11

    return input_sentence,output_sentence,error_position

def similar_wrong(input_sentence,output_sentence,error_position):
    pre3 = ['ngh']
    pre2 = ['gh', 'ng']
    pre1 = ['g', 'c', 'q', 'k']

    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and word[:1] in pre1 or word[:2] in pre2 or word[:3] in pre3]
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        if word[:3] in pre3:
            word = choice(like_pronunciation2[word[:3]]) + word[3:]
        elif word[:2] in pre2:
            word = choice(like_pronunciation2[word[:2]]) + word[2:]
        elif word[:1] in pre1:
            word = choice(like_pronunciation2[word[:1]]) + word[1:]
        input_sentence[i]=word
        error_position[i] = 12

    return input_sentence,output_sentence,error_position

def wrong_position_diacritic(input_sentence,output_sentence,error_position):
    sub2 = ['óa', 'oá', 'òa','oà', 'ỏa', 'oả', 'õa', 'oã', 'ọa', 'oạ',\
            'áo', 'aó', 'ào','aò', 'ảo', 'aỏ', 'ão', 'aõ', 'ạo', 'aọ',\
            'éo', 'eó', 'èo','eò', 'ẻo', 'eỏ', 'ẽo', 'eõ', 'ẹo', 'eọ',\
            'óe', 'oé', 'òe','oè', 'ỏe', 'oẻ', 'õe', 'oẽ', 'ọe', 'oẹ',\
            'ái', 'aí', 'ài','aì', 'ải', 'aỉ', 'ãi', 'aĩ', 'ại', 'aị',\
            'áy', 'aý', 'ày','aỳ', 'ảy', 'aỷ', 'ãy', 'aỹ', 'ạy', 'aỵ',\
            'ói', 'oí', 'òi','oì', 'ỏi', 'oỉ', 'õi', 'oĩ', 'ọi', 'oị'] # convert ve khong dau 

    change_dict = {'óa': 'oá', 'òa':'oà', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',\
                    'oá': 'óa', 'oà':'òa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',\
                    'áo': 'aó', 'ào':'aò', 'ảo': 'aỏ', 'ão': 'aõ', 'ạo': 'aọ',\
                    'aó': 'áo', 'aò':'ào', 'aỏ': 'ảo', 'aõ': 'ão', 'aọ': 'ạo',\
                    'éo': 'eó', 'èo':'eò', 'ẻo': 'eỏ', 'ẽo': 'eõ', 'ẹo': 'eọ',\
                    'eó': 'éo', 'eò':'èo', 'eỏ': 'ẻo', 'eõ': 'ẽo', 'eọ': 'ẹo',\
                    'óe': 'oé', 'òe':'oè', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',\
                    'oé': 'óe', 'oè':'òe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe',\
                    'ái': 'aí', 'ài':'aì', 'ải': 'aỉ', 'ãi': 'aĩ', 'ại': 'aị',\
                    'aí': 'ái', 'aì':'ài', 'aỉ': 'ải', 'aĩ': 'ãi', 'aị': 'ại',\
                    'ói': 'oí', 'òi':'oì', 'ỏi': 'oỉ', 'õi': 'oĩ', 'ọi': 'oị',\
                    'oí': 'ói', 'oì':'òi', 'oỉ': 'ỏi', 'oĩ': 'õi', 'oị': 'ọi',\
                    'áy': 'aý', 'ày':'aỳ', 'ảy': 'aỷ', 'ãy': 'aỹ', 'ạy': 'aỵ',\
                    'aý': 'áy', 'aỳ':'ày', 'aỷ': 'ảy', 'aỹ': 'ãy', 'aỵ':'ạy'
                    }
    maybe_fix = []
    for i,word in enumerate(input_sentence):
        if error_position[i] == 0:
            sub = [s for s in sub2 if s in word]
            if len(sub)>0:
                sub = choice(sub)
                maybe_fix.append([i,sub])
    if len(maybe_fix)>0:
        a = choice(range(len(maybe_fix)))
        i = maybe_fix[a][0]
        sub = maybe_fix[a][1]
        word = input_sentence[i]
        word = word.replace(sub,change_dict[sub])
        input_sentence[i] = word
        error_position[i] = 13
    
    return input_sentence,output_sentence,error_position
    
def add_char(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>1]
    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        j = random.randint(0, len(word) - 1)
        new_char = choice(list(NO_DIACRITIC_VN_CHAR))
        word = word[:j] + new_char + word[j+1:]
        input_sentence[i] = word
        error_position[i] = 14

    return input_sentence,output_sentence,error_position

def change_diacritic(input_sentence, output_sentence, error_position):
    def get_diacritic(word):
        for i, char in enumerate(word):
            if char in HUYEN:
                return DIACRITIC_LIST[2], i
            elif char in SAC:
                return DIACRITIC_LIST[1], i
            elif char in HOI:
                return DIACRITIC_LIST[3], i
            elif char in NGA:
                return DIACRITIC_LIST[4], i
            elif char in NANG:
                return DIACRITIC_LIST[5], i
        for i, char in enumerate(reversed(word)):
            if char in ["a", "e", "i", "o", "u", "y", 'â', 'ă', 'ê', 'ô', 'ơ', 'ư']:
                # print(f'Char is {char}')
                return DIACRITIC_LIST[0], len(word) - i - 1

    maybe_fix = [i for i, e in enumerate(error_position) if e==0]
    if len(maybe_fix) > 0:

        i = choice(maybe_fix)
        word = input_sentence[i]
        # print(f'Word is {word}')
        if get_diacritic(word):
            diacritic, pos = get_diacritic(word)
        else:
            return input_sentence, output_sentence, error_position
        # print(f'Letter is {word[pos]}')
        # print(f'Dau la {diacritic}')
        pool_list = [d for d in DIACRITIC_LIST if d != diacritic]
        new_diacritic = choice(pool_list)

        for key, val in bo_dau.items():
            if word[pos] in val:
                # print('ITS IN HEREEEEE')
                letter = key
                break
        
        letter = bo_dau[letter][DIACRITIC_LIST.index(new_diacritic)]

        word = word[:pos] + letter + word[pos+1:]
        input_sentence[i] = word
        error_position[i] = 15
    return input_sentence, output_sentence, error_position

def replace_component(input_sentence, output_sentence, error_position):
    maybe_fix = [i for i, e in enumerate(error_position) if e==0]
    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        word_components = []
        component_start_pos = 0
        switch = False
        for ind, char in enumerate(word):
            if not switch:
                if char in AEIOUY_VN:
                    if ind != 0:
                        word_components.append(word[:ind])
                    if ind == len(word) - 1:
                        word_components.append(word[ind])

                    component_start_pos = ind
                    switch = True
            else:
                if char not in AEIOUY_VN:
                    word_components.append(word[component_start_pos:ind])
                    word_components.append(word[ind:])
                    break
                elif ind == len(word) - 1:
                    word_components.append(word[component_start_pos:])
        
        if not switch:
            return input_sentence, output_sentence, error_position
        # print(f'Word: {word}')
        # print(word_components)
        replace_pos = random.randint(0, len(word_components))
        if word_components[replace_pos] in CONSONANT:
            word_components[replace_pos] = choice(CONSONANT)
        elif word_components[replace_pos] in VOWEL:
            word_components[replace_pos] = choice(VOWEL)

        input_sentence[i] = ''.join(word_components)
        error_position[i] = 16
    return input_sentence, output_sentence, error_position

def remove_diacritic(word):
    for i, char in enumerate(word):
        for key, val in bo_dau.items():
            if char in val:
                new_word = word[:i] + key + word[i+1:]
                return new_word
    return word

if __name__ == "__main__":
    a = []
    a.append(None)
    input_sentence = ['hlv']
    output_sentence = ['hlv']
    error_position = [0]
    print(add_noise_sentence(input_sentence,output_sentence,error_position,15))
    # c = 'đ'
    # dau = [dau_cau[i] for i in dau_cau if c in dau_cau[i]]
    # print(choice(dau[0]))