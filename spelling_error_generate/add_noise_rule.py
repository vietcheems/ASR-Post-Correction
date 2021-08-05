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
    10 nga va hoi

    11 cac am vi o dau co cach phat am giong nhau trong mot so truong hop, vi du c q k
    12 sai vi tri dau va thay doi chu cai voi to hop cac dau neu am tiet co 1 nguyen am
"""
import numpy as np 
from numpy.random import choice
import numpy.random as random
import unidecode
import copy
from alphabet import *
MAX_LOOP = 50

def add_noise_sentence(input_sentence,output_sentence,error_position,type_noise):
    input_sentence = copy.deepcopy(input_sentence)
    output_sentence = copy.deepcopy(output_sentence)
    if type_noise == 1:
        return swap_char(input_sentence,output_sentence,error_position)
    elif type_noise == 2:
        return thieu_dau_phu(input_sentence,output_sentence,error_position)
    elif type_noise == 3:
        return g_to_ng_error(input_sentence,output_sentence,error_position)
    elif type_noise == 4:
        return y_or_i_error(input_sentence,output_sentence,error_position)
    elif type_noise == 5:
        return wrong_position_diacritic(input_sentence,output_sentence,error_position)
    elif type_noise == 6:
        return c_q_k_wrong(input_sentence,output_sentence,error_position)
    elif type_noise == 7:
        return f_to_ph_error(input_sentence,output_sentence,error_position)  #
    elif type_noise == 8:
        return repeat_vowel(input_sentence,output_sentence,error_position)

def g_to_ng_error(input_sentence,output_sentence,error_position):

    assert len(input_sentence) == len(output_sentence)
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and word[-2:]=='ng']
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        word = word[:-2]+'g'
        input_sentence[i]=word
        error_position[i] = 3

    return input_sentence,output_sentence,error_position
def thieu_dau_phu(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and ('ie'in unidecode.unidecode(word) or 'ye' in unidecode.unidecode(word)or 'uu' in unidecode.unidecode(word)or 'uo' in unidecode.unidecode(word))]
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        no_tone_word = ''.join([get_no_tone[i] for i in word])
        if 'yê' in no_tone_word or 'iê' in no_tone_word:
            x = no_tone_word.find('ê')
            tone = bo_dau[get_no_tone[word[x]]].find(word[x])
            word = word[:x]+bo_dau['e'][tone]+ word[x+1:]
            input_sentence[i]=word
            error_position[i] = 2
        elif 'ươ' in no_tone_word:
            ch = np.random.choice(2)
            if ch:
                x = no_tone_word.find('ơ')
                tone = bo_dau[get_no_tone[word[x]]].find(word[x])
                word = word[:x]+bo_dau['o'][tone]+ word[x+1:]
                input_sentence[i]=word
                error_position[i] = 2
            else:
                input_sentence[i]=word.replace('ư','u')
                error_position[i] = 2                
        elif 'ưu' in no_tone_word:
            x = no_tone_word.find('ư')
            tone = bo_dau[get_no_tone[word[x]]].find(word[x])
            word = word[:x]+bo_dau['u'][tone]+ word[x+1:]
            input_sentence[i]=word
            error_position[i] = 2
    return input_sentence,output_sentence,error_position
def y_or_i_error(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and ('i'in unidecode.unidecode(word) or 'y' in unidecode.unidecode(word))]
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        fix_word = ''.join([get_no_tone[c] for c in word])

        dau = 0
        for j,c in enumerate(fix_word):
            if c in bo_dau:
                dau= max(dau,bo_dau[c].find(word[j]))
        if 'i' in fix_word:
            fix_word = fix_word.replace('i','y')
        else:

            fix_word = fix_word.replace('y','i')

        x = max(fix_word.find('y'),fix_word.find('i'))
        if fix_word.find('y')<fix_word.find('i'):
            fix_word = fix_word[:x]+ bo_dau['i'][dau] + fix_word[x+1:]
        else:
            fix_word = fix_word[:x]+ bo_dau['y'][dau] + fix_word[x+1:]

        input_sentence[i]=''.join(fix_word)
        error_position[i] = 4
    
    return input_sentence,output_sentence,error_position
def f_to_ph_error(input_sentence,output_sentence,error_position):
   
    assert len(input_sentence) == len(output_sentence)
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and word[:2]=='ph']
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        word = 'f'+word[2:]
        input_sentence[i]=word
        error_position[i] = 7

    return input_sentence,output_sentence,error_position

def swap_char(input_sentence,output_sentence,error_position):
    assert len(input_sentence) == len(output_sentence)
    diphones = ['hn','hc','ig','hg','hk','gn','ht','rt','uq','hp','êi','êy','ôu','ơư','ei','ey','oư','aư','âư']
    diphones_fix = ['nh','ch','gi','gh','kh','ng','th','tr','qu','ph','iê','yê','uô','ươ','ie','ye','ưo','ưa','ưâ']
    maybe_fix = [i for i,e in enumerate(error_position) if e==0 and len(input_sentence[i])>2 and any(d in input_sentence[i] for d in diphones_fix)] 

    if len(maybe_fix) > 0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        for j,d in enumerate(diphones_fix):
            if d in word:
                word = word.replace(d,diphones[j],1)
                break
                
        input_sentence[i] = word
        error_position[i] = 1

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



def c_q_k_wrong(input_sentence,output_sentence,error_position):
    pre3 = ['ngh']
    pre2 = ['gh', 'ng']
    pre1 = ['g', 'c', 'k']

    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and len(word)>1 and word[:1] in pre1 or word[:2] in pre2 or word[:3] in pre3]
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        if word[:3] in pre3:
            word = choice(like_pronunciation2[word[:3]]) + word[3:]
            input_sentence[i]=word
            error_position[i] = 6
        elif word[:2] in pre2:
            word = choice(like_pronunciation2[word[:2]]) + word[2:]
            input_sentence[i]=word
            error_position[i] = 6
        elif word[:1] in pre1 and unidecode.unidecode(word[1]) in 'eiouay' and word[:2]!='gi':
            word = choice(like_pronunciation2[word[:1]]) + word[1:]
            input_sentence[i]=word
            error_position[i] = 6

    return input_sentence,output_sentence,error_position
def num_dia(word):
    dem=0
    for c in word:
        if unidecode.unidecode(c) in 'aeiouy':
            dem+=1
    return dem
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
    maybe_fix = [i for i,word in enumerate(output_sentence) if error_position[i]==0 and num_dia(word)>=2]
    # no_tone_word
    if len(maybe_fix)>0:
        i = choice(maybe_fix)
        word = input_sentence[i]
        tone_pos = []
        tone =0
        for j, c  in enumerate(word):
            if get_no_tone[c] in bo_dau:
                x = bo_dau[get_no_tone[c]].find(c)
                if x !=0:
                    tone = x
                else:
                    tone_pos.append(j)
        if len(tone_pos)!= 0 and tone!=0:
            x = choice(tone_pos)
            word = ''.join([get_no_tone[c] for c in word])
            word = word[:x] + bo_dau[word[x]][tone]+ word[x+1:]
            input_sentence[i] = word
            error_position[i] = 5
    
    return input_sentence,output_sentence,error_position

if __name__ == "__main__":
    input_sentence = ['xin','chào','cáng','bãn','nhanh']
    output_sentence = ['xin','chào','cáng','bãn','nhanh']
    error_position = [0,0,0,0,0]
    print(add_noise_sentence(input_sentence,output_sentence,error_position,1))
    # c = 'đ'
    # dau = [dau_cau[i] for i in dau_cau if c in dau_cau[i]]
    # print(choice(dau[0]))