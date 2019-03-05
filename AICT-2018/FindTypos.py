import math
from nltk.corpus import wordnet
import nltk
from collections import Counter, defaultdict
import sys
from nltk.stem import *
from nltk.stem.porter import PorterStemmer
import difflib
import enchant
import re
from collections import Counter
from math import log
dic = enchant.DictWithPWL("en_US", "ComputerTerms.txt")
st = PorterStemmer()
# directory = r'c:\TypoDetectionDatasets\\'
directory = '.\\'
Datasets = ["Android","Eclipse","Mozilla","Openoffice"]

def arrToStr(arr, sep = ','):
    return sep.join([str(a) for a in arr])

def my_tokenize(str):
    tokens = nltk.word_tokenize(str)
    for k in range(len(tokens)):
        if tokens[k] == "n't":
            tokens[k] = "not"
        elif tokens[k] == "'m":
             tokens[k] = "am"
        elif tokens[k] == "'s":
             tokens[k] = "is"
    return tokens

def FindTypos(strInput):
    tokens = my_tokenize(strInput.replace('_',' ').strip())
    out = []
    for t in tokens:
        s = t
        if not dic.check(s):
            try:
                if not dic.check(st.stem(s)):
                    out.append(s)
            except:
                out.append(s)
    lo=len(out)
    out=list(set(out))
    return (len(tokens),lo,len(out),out)

def Process(DS,InputTextFileName, OutputFileName,StatFileName,typo):
    fout = open(OutputFileName,'w')
    fstat = open(StatFileName,'w')
    fstat.write('DS,Id,LenT,TypoT,UTT,LenD,TypoD,UTD,LenB,TypoB,UTB\n')
    fout.write('Id,Place,Word\n')
    i=0
    with open(InputTextFileName,'r') as fin:
        next(fin)
        for l in fin:
##            l = l.decode('utf-8')
            s = l.strip().split(',')
            s = s[0:3]
            s[0] = int(s[0])
            tt = FindTypos(s[1])
            td = FindTypos(s[2])
            fstat.write(str.format('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n'
                        ,DS,s[0],tt[0],tt[1],tt[2]
                        ,td[0],td[1],td[2],tt[0]+td[0],tt[1]+td[1],tt[2]+td[2]))
            for t in tt[3]:
                fout.write(arrToStr([s[0],'T',t])+"\n")
                typo.add(t)
            for t in td[3]:
                fout.write(arrToStr([s[0],'D',t])+"\n")
                typo.add(t)
            i+=1
            if i%5000==0:
                print(i)
    fout.close()
    fstat.close()
    
if __name__ == "__main__":
    totaltypo = set()
    for i in range(len(Datasets)):
        d = Datasets[i]
        print(d)
        typo=set()
        Process( d
                ,str.format('{0}{1}1{2}TitleDescriptionDataset.txt',directory,i+1,d)
                ,str.format('{0}{1}2{2}Typos.txt',directory,i+1,d)
                ,str.format('{0}{1}4{2}Statistic.txt',directory,i+1,d)
                ,typo)
        fout = open(str.format('{0}{1}3{2}UniqueTypos.txt',directory,i+1,d),'w')
        typol=list(typo)
        typol.sort()
        fout.write('\n'.join(typol))
        fout.close()
        totaltypo.update(typo)
    fout = open(directory+'AllUniqueTypo.txt','w')
    fout.write('\n'.join(list(totaltypo)))
    fout.close()
