dicFile = 'Words.fa.txt'

import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

clspy = lambda: print('\n'*100)
def clspy():
    for c in range(100):
        print('')
from itertools import permutations

#ی
def ReadWordList(file):
    wl = []
    with open(file,'r',encoding='UTF8') as f:
        wl = [c.strip() for c in f.read().split('\n')]
    wl.sort()
    return wl

def BuiltNMT(wl):
    nc = 0
    nmt = [{},False]
    for w in wl:
        n = nmt
        for ci in range(len(w)):
            c = w[ci]
            if c not in n[0]:
                n[0][c] = [{},False]
                nc += 1
            n = n[0][c]
            if ci==len(w)-1:
                n[1]=True
    return nmt,nc

def SearchNMT(nmt,w):
    n = nmt
    for c in w:
        if c not in n[0]:
            return False
        n = n[0][c]
    return n[1]

def GetAllComb(nmt,comb,k):
    scomb = set(comb)
    allcomb = [['',0,nmt]]
    res = []
    while len(allcomb)!=0:
        n = allcomb[0]
        allcomb = allcomb[1:]
        if n[1]>k:
            continue
        if n[2][1]:
            res.append(n[0])
        for c in list(n[2][0].keys() & comb):
            allcomb.append([n[0]+c,n[1]+1,n[2][0][c]])
    return res
    

wl = ReadWordList(dicFile)
nmt,nc = BuiltNMT(wl)

def test2(s,k=6):
    if 'ا' in s: s+='آأ'
    if 'آ' in s: s+='اأ'
    if 'ي' in s: s+='یئ'
    if 'ی' in s: s+='يئ'
    out = GetAllComb(nmt,s,k)
    return set(out)
##    out.sort(key=lambda c:(len(c),c))    
##    clspy()
##    print('--------------------')
##    for w in out:
##        print(w,"*"*2*(len(w)+1),'  ',len(w))

##test2('ابسن')

global k,chars,out
def check12(w):
    global k,chars,out,nmt
##    len(w)>=2 and 
    if SearchNMT(nmt,w):
        out.append(w)
##        print(w,"*"*2*(len(w)+1),'  ',len(w))
    if len(w)>k: return
    for c in chars:
        wn = w+c
        if len(wn)<=k: check12(wn)

def test12(s,kk=6):
    global k,chars,out
    out = []
    k = kk
    if 'ا' in s: s+='آأ'
    if 'آ' in s: s+='اأ'
    if 'ي' in s: s+='یئ'
    if 'ی' in s: s+='يئ'
    chars = list(set(s))
    chars.sort()
    check12('')
    return set(out)
##    out.sort(key=lambda c:(len(c),c))
##    clspy()
##    print('--------------------')
##    for w in out:
##        print(w,"*"*2*(len(w)+1),'  ',len(w))
    
##test('')


import enchant
dic1 = enchant.DictWithPWL("en_US", dicFile)

global k,chars,out
def check11(w):
    global k,chars,out
##    if len(w)>=3 and  (dic1.check(w) or dic2.check(w) or dic3.check(w) or dic4.check(w)):
    # len(w)>=3 and 
    if len(w)>=2 and dic1.check(w):
        out.append(w)
##        print(w,"*"*(len(w)+1),'  ',len(w))
    if len(w)>k:
        return
    for c in chars:
        wn = w+c
        if len(wn)<=k: check11(wn)
##        if len(w)<=6:
##            check(w)
    


def test11(s,kk=6):
    global k,chars,out
    out = []
    k = kk
    if 'ا' in s:
        s+='آأ'
    if 'آ' in s:
        s+='اأ'
    if 'ي' in s:
        s+='یئ'
    if 'ی' in s:
        s+='يئ'
##    if 'ک' in s:
##        s+='ک'
    chars = list(set(s))
    chars.sort()
    check11('')
    return out
##    out.sort(key=lambda c:(len(c),c))
##    clspy()
##    print('--------------------')
##    for w in out:
##        print(w,"*"*(len(w)+1),'  ',len(w))
    
##test('')
def test0(chars):
    out = []
    for r in range(2,6+1):
        per_chars = permutations(chars, r)
        for pc in per_chars:
            w = ''.join(pc)
            if dic1.check(w):
                out.append(w)
##                print(w)
    return set(out)

import time

def ti(f,c):
    s = time.time()
    o = f(c)
    e = time.time()
    return e-s, len(o), o

def printTest(s):
    o0 = ti(test0,s)
    o11 = ti(test11,s)
    o12 = ti(test12,s)
    o2 = ti(test2,s)
    print(s,o0[0],o0[1],o11[0],o11[1],o12[0],o12[1],o2[0],o2[1],sep=',')

print('nc=',nc)

printTest('آسنترد')
##printTest('دسروه')
##printTest('مشهود')
##printTest('انحدو')
##printTest('طوکير')
##printTest('پانيز')
##printTest('قريچب')
##
##printTest('ايلتجگ')
##printTest('ارمنسه')
##printTest('مسلوه')
##printTest('ابتلفن')
##printTest('درسنيف')
##
##printTest('قوانيپل')
##printTest('امرتون')
##printTest('انسيچ')
##printTest('ناوديقب')
##printTest('گلايمن')
