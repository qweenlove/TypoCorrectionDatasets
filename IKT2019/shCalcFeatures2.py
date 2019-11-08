# coding=utf8
#
# Copyright (c) 2012, Frane Saric
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
#   * If this software or its derivative is used to produce an academic
# publication, you are required to cite this work by using the citation
# provided on "http://takelab.fer.hr/sts".
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from random import shuffle
import math
from nltk.corpus import wordnet
import nltk
from collections import Counter, defaultdict
import sys
import re
import numpy
from numpy.linalg import norm
import datetime
import sys

def arrToStr(a, sep = ','):
##    formatStr = '%f'+sep
##    return ' '.join(formatStr % (x) for x in a)
    x=""
    for i in a:
        x = x + str(i)+sep
    return x
class Sim:
    def __init__(self, words, vectors):
        self.word_to_idx = {a: b for b, a in
                            enumerate(w.strip() for w in open(words))}
        self.mat = numpy.loadtxt(vectors)

    def bow_vec(self, b):
        vec = numpy.zeros(self.mat.shape[1])
        for k, v in b.items():
            idx = self.word_to_idx.get(k, -1)
            if idx >= 0:
                vec += self.mat[idx] / (norm(self.mat[idx]) + 1e-8) * v
        return vec

    def calc(self, b1, b2):
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        return abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8))

stopwords = nltk.corpus.stopwords.words('english')
# set([
# "i", "a", "about", "an", "are", "as", "at", "be", "by", "for", "from",
# "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
# "was", "what", "when", "where", "who", "will", "with", "the", "'s", "did",
# "have", "has", "had", "were", "'ll"
# ])

nyt_sim = Sim('nyt_words.txt', 'nyt_word_vectors.txt')
wiki_sim = Sim('wikipedia_words.txt', 'wikipedia_word_vectors.txt')

def fix_compounds(a, b):
    sb = set(x.lower() for x in b)

    a_fix = []
    la = len(a)
    i = 0
    while i < la:
        if i + 1 < la:
            comb = a[i] + a[i + 1]
            if comb.lower() in sb:
                a_fix.append(a[i] + a[i + 1])
                i += 2
                continue
        a_fix.append(a[i])
        i += 1
    return a_fix
def my_tokenize(str):
    tokens = nltk.word_tokenize(str)
    for k in range(len(tokens)):
        if tokens[k] == "n't":
            tokens[k] = "not"
        elif tokens[k] == "'m":
             tokens[k] = "am"
    return tokens
def load_data(path):
    bugs = {}
    r1 = re.compile(r'\<([^ ]+)\>')
    r2 = re.compile(r'\$US(\d)')
    with open(path,'rb') as f:
        next(f)
        for l in f:
            l = l.decode('utf-8')
            l = l.replace(u'’', "'")
            l = l.replace(u'``', '"')
            l = l.replace(u"''", '"')
            l = l.replace(u"—", '--')
            l = l.replace(u"–", '--')
            l = l.replace(u"´", "'")
            l = l.replace(u"-", " ")
            l = l.replace(u"/", " ")
            l = r1.sub(r'\1', l)
            l = r2.sub(r'$\1', l)
            s = l.strip().split(',')
##            if s[0] in ('9464','30717','19235'):
##                print(s)
            s = s[0:3]
            s.append(my_tokenize(s[1]))
            s.append(my_tokenize(s[2]))
            bugs[int(s[0])]=s
    return bugs
def load_wweight_table(path):
    lines = open(path).readlines()
    wweight = defaultdict(float)
    if not len(lines):
        return (wweight, 0.)
    totfreq = int(lines[0])
    for l in lines[1:]:
        w, freq = l.split()
        freq = float(freq)
        if freq < 10:
            continue
        wweight[w] = math.log(totfreq / freq)

    return wweight

wweight = load_wweight_table('word-frequencies.txt')
minwweight = min(wweight.values())

def len_compress(l):
    return math.log(1. + l)

to_wordnet_tag = {
        'NN':wordnet.NOUN,
        'JJ':wordnet.ADJ,
        'VB':wordnet.VERB,
        'RB':wordnet.ADV
    }

word_matcher = re.compile('[^0-9,.(=)\[\]/_`]+$')
def is_word(w):
    return word_matcher.match(w) is not None

def get_locase_words(spos):
    return [x[0].lower() for x in spos
            if is_word(x[0])]

def make_ngrams(l, n):
    rez = [l[i:(-n + i + 1)] for i in range(n - 1)]
    rez.append(l[n - 1:])
    return list(zip(*rez))

def dist_sim(sim, la, lb):
    wa = Counter(la)
    wb = Counter(lb)
    d1 = {x:1 for x in wa}
    d2 = {x:1 for x in wb}
    return sim.calc(d1, d2)

def weighted_dist_sim(sim, lca, lcb):
    wa = Counter(lca)
    wb = Counter(lcb)
    wa = {x: wweight[x] * wa[x] for x in wa}
    wb = {x: wweight[x] * wb[x] for x in wb}
    return sim.calc(wa, wb)

def weighted_word_match(lca, lcb):
    wa = Counter(lca)
    wb = Counter(lcb)
    wsuma = sum(wweight[w] * wa[w] for w in wa)
    wsumb = sum(wweight[w] * wb[w] for w in wb)
    wsum = 0.

    for w in wa:
        wd = min(wa[w], wb[w])
        wsum += wweight[w] * wd
    p = 0.
    r = 0.
    if wsuma > 0 and wsum > 0:
        p = wsum / wsuma
    if wsumb > 0 and wsum > 0:
        r = wsum / wsumb
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1

wpathsimcache = {}
def wpathsim(a, b):
    if a > b:
        b, a = a, b
    p = (a, b)
    if p in wpathsimcache:
        return wpathsimcache[p]
    if a == b:
        wpathsimcache[p] = 1.
        return 1.
    sa = wordnet.synsets(a)
    sb = wordnet.synsets(b)
    #print(sa[0],sb[0],sa[0].path_similarity(sb[0]), type(sa[0].path_similarity(sb[0])))
    mx = max([x for x in [wa.path_similarity(wb)
              for wa in sa
              for wb in sb
              ] if x != None] + [0.])
    wpathsimcache[p] = mx
    return mx

def calc_wn_prec(lema, lemb):
    rez = 0.
    for a in lema:
        ms = 0.
        for b in lemb:
            ms = max(ms, wpathsim(a, b))
        rez += ms
    return rez / len(lema)

def wn_sim_match(lema, lemb):
    f1 = 1.
    p = 0.
    r = 0.
    if len(lema) > 0 and len(lemb) > 0:
        p = calc_wn_prec(lema, lemb)
        r = calc_wn_prec(lemb, lema)
        f1 = 2. * p * r / (p + r) if p + r > 0 else 0.
    return f1

def ngram_match(sa, sb, n):
    nga = make_ngrams(sa, n)
    ngb = make_ngrams(sb, n)
    matches = 0
    c1 = Counter(nga)
    for ng in ngb:
        if c1[ng] > 0:
            c1[ng] -= 1
            matches += 1
    p = 0.
    r = 0.
    f1 = 1.
    if len(nga) > 0 and len(ngb) > 0:
        p = matches / float(len(nga))
        r = matches / float(len(ngb))
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1

def get_lemmatized_words(sa):
    rez = []
    for w, wpos in sa:
        w = w.lower()
        if w in stopwords or not is_word(w):
            continue
        wtag = to_wordnet_tag.get(wpos[:2])
        if wtag is None:
            wlem = w
        else:
            wlem = wordnet.morphy(w, wtag) or w
        rez.append(wlem)
    return rez

def is_stock_tick(w):
    return w[0] == '.' and len(w) > 1 and w[1:].isupper()

def stocks_matches(sa, sb):
    ca = set(x[0] for x in sa if is_stock_tick(x[0]))
    cb = set(x[0] for x in sb if is_stock_tick(x[0]))
    isect = len(ca.intersection(cb))
    la = len(ca)
    lb = len(cb)

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f)

def case_matches(sa, sb):
    ca = set(x[0] for x in sa[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    cb = set(x[0] for x in sb[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    la = len(ca)
    lb = len(cb)
    isect = len(ca.intersection(cb))

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f)

risnum = re.compile(r'^[0-9,./-]+$')
rhasdigit = re.compile(r'[0-9]')

def match_number(xa, xb):
    if xa == xb:
        return True
    xa = xa.replace(',', '')
    xb = xb.replace(',', '')

    try:
        va = int(float(xa))
        vb = int(float(xb))
        if (va == 0 or vb == 0) and va != vb:
            return False
        fxa = float(xa)
        fxb = float(xb)
        if abs(fxa - fxb) > 1:
            return False
        diga = xa.find('.')
        digb = xb.find('.')
        diga = 0 if diga == -1 else len(xa) - diga - 1
        digb = 0 if digb == -1 else len(xb) - digb - 1
        if diga > 0 and digb > 0 and va != vb:
            return False
        dmin = min(diga, digb)
        if dmin == 0:
            if abs(round(fxa, 0) - round(fxb, 0)) < 1e-5:
                return True
            return va == vb
        return abs(round(fxa, dmin) - round(fxb, dmin)) < 1e-5
    except:
        pass

    return False

def number_features(sa, sb):
    numa = set(x[0] for x in sa if risnum.match(x[0]) and
            rhasdigit.match(x[0]))
    numb = set(x[0] for x in sb if risnum.match(x[0]) and
            rhasdigit.match(x[0]))
    isect = 0
    for na in numa:
        if na in numb:
            isect += 1
            continue
        for nb in numb:
            if match_number(na, nb):
                isect += 1
                break

    la, lb = len(numa), len(numb)

    f = 1.
    subset = 0.
    if la + lb > 0:
        if isect == la or isect == lb:
            subset = 1.
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2. * p * r / (p + r)
        else:
            f = 0.
    return (len_compress(la + lb), f, subset)

def relative_len_difference(lca, lcb):
    la, lb = len(lca), len(lcb)
    return abs(la - lb) / float(max(la, lb) + 1e-5)

def relative_ic_difference(lca, lcb):
    #wa = sum(wweight[x] for x in lca)
    #wb = sum(wweight[x] for x in lcb)
    wa = sum(max(0., wweight[x] - minwweight) for x in lca)
    wb = sum(max(0., wweight[x] - minwweight) for x in lcb)
    return abs(wa - wb) / (max(wa, wb) + 1e-5)

def calc_features(sa, sb):
    olca = get_locase_words(sa)
    olcb = get_locase_words(sb)
    lca = [w for w in olca if w not in stopwords]
    lcb = [w for w in olcb if w not in stopwords]
    lema = get_lemmatized_words(sa)
    lemb = get_lemmatized_words(sb)
    f = []
    f += number_features(sa, sb)
    f += case_matches(sa, sb)
    f += stocks_matches(sa, sb)
    f += [
##            ngram_match(lca, lcb, 1),
##            ngram_match(lca, lcb, 2),
##            ngram_match(lca, lcb, 3),
            ngram_match(lema, lemb, 1),
            ngram_match(lema, lemb, 2),
##            ngram_match(lema, lemb, 3),
            wn_sim_match(lema, lemb),
##            wn_sim_match(lca, lcb),
##            weighted_word_match(olca, olcb),
##            weighted_word_match(lca, lcb),
##            dist_sim(nyt_sim, lca, lcb),
##            dist_sim(wiki_sim, lca, lcb),
##            weighted_dist_sim(nyt_sim, lca, lcb),
##            weighted_dist_sim(wiki_sim, lca, lcb),
            weighted_word_match(lema, lemb),
            dist_sim(nyt_sim, lema, lemb),
            dist_sim(wiki_sim, lema, lemb),
            weighted_dist_sim(nyt_sim, lema, lemb),
            weighted_dist_sim(wiki_sim, lema, lemb),
            relative_len_difference(lca, lcb),
##            relative_ic_difference(olca, olcb)
        ]
    return f
    
def make_str_features(sa,sb):
    sa, sb = fix_compounds(sa, sb), fix_compounds(sb, sa)
    sp = (nltk.pos_tag(sa), nltk.pos_tag(sb))
    return ' '.join('%f,' % (x) for x in calc_features(*sp))[0:-1]

def readFileTitDes(file):
    #Id,Title,Description
    res = {}
    with open(file,'r') as f:
        next(f)
        for l in f:
            ps = [x.strip() for x in l.split(',')]
            res[ps[0]]=ps[1:]
    return res

def readFileCorrectedTerms(file):
    #Word,Corrected,Connected,Additional,Removal,Subs,Undefined,??????
    res = {}
    with open(file,'r') as f:
        next(f)
        for l in f:
            ps = [x.strip() for x in l.split(',')]
            for i in range(2,len(ps)):
                if ps[i].isdigit():
                    ps[i] = int(ps[i])
            res[ps[0]]=ps[1:]
    return res

def readFileIdLbl(file):
    #Bug1,Bug2,prod,comp,type,priorA,versA,date,id,priorS,versS,,cosine_similarity,class
    res = {}
    with open(file,'r') as f:
        next(f)
        for l in f:
            ps = [x.strip() for x in l.split(',')]
            ps[0],ps[1]=int(ps[0]),int(ps[1])
            if ps[-1]=='dup':
                n,x=min(ps[0:2]),max(ps[0:2])
                if n not in res: res[n]=[]
                if x not in res: res[x]=[]
                res[n].append(x)
##                if ps[1] not in res: res[ps[1]]=[]
##                if ps[1] not in res[ps[0]]: res[ps[0]].append(ps[1])
##                if ps[0] not in res[ps[1]]: res[ps[1]].append(ps[0])
    return res

def readFileTypos(file,cts):
    #Id,Place,Word
    res = {}
    with open(file,'r') as f:
        next(f)
        for l in f:
            ps = [x.strip() for x in l.split(',')]
            if ps[2] not in cts: continue
            ps[0] = int(ps[0])
            if ps[0] not in res: res[ps[0]]={'T':[],'D':[]}
            res[ps[0]][ps[1]].append(ps[2])
    return res

def readFileFields(file):
    #BugID,MasterID,MergeID,Product,Component,Type,Priority,PriorityNumber,Version,VersionNumber,OpenDate,CloseDate,Status,Stars,Title,Description,Summary,crypto,general,java,networking
    res = {}
    with open(file,'r',encoding="utf8") as f:
        next(f)
        for l in f:
            ps = [x.strip().lower() for x in l.split(',')]
            if ps[12]=='': continue
            if len(ps)!=21:
                print('len! ', ps[0])
                continue
            ps[0],ps[1],ps[2],ps[7],ps[9]=int(ps[0]),int(ps[1]),int(ps[2]),float(ps[7]),float(ps[9])
            ps[10] = datetime.datetime.strptime(ps[10], '%d %b %Y %H:%M:%S GMT')
            ps[11] = datetime.datetime.strptime(ps[11], '%d %b %Y %H:%M:%S GMT')
            ps[14:18]=[float(x) for x in ps[14:18]]
            res[ps[0]]=ps #[1:]
    return res

def cmpAdd(a,b):
    return 1.0/(1+(a-b)) if 1+(a-b)!=0 else 0

def cmpSub(a,b):
    return 1.0/(1-(a-b)) if 1-(a-b)!=0 else 0

def isEqual(a,b):
    return 1 if a.lower()==b.lower() else 0

def lv(a):
    return sum([x**2 for x in a])**0.5

def cos_sim(a,b):
    lvab = lv(a)*lv(b)
    return sum([a[i]*b[i] for i in range(len(a))])/lvab if lvab!=0 else 0

def rmv(txt,brtn):
    for t in brtn:
        txt = txt.replace(t,' ')
    return txt

def rep(txt,brtn,cts):
    for t in brtn:
        if t in cts:
            txt = txt.replace(t,cts[t][0])
    return txt

def calcAllFeat(brfs,brt,cts,br1,br2,ds):
    fs = [br1,br2,ds]
    b1,b2 = brfs[br1],brfs[br2]
    fs.append(isEqual(b1[3],b2[3])) # product
    fs.append(isEqual(b1[4],b2[4])) # component
    fs.append(isEqual(b1[5],b2[5])) # type
    fs.append(cmpAdd(b1[7],b2[7])) # PrioAdd
    fs.append(cmpSub(b1[7],b2[7])) # PrioSub
    fs.append(cmpAdd(b1[9],b2[9])) # VerAdd
    fs.append(cmpSub(b1[9],b2[9])) # VerSub
    fs.append((b1[10]-b2[11]).total_seconds()) # date
    fs.append(abs((b1[10]-b2[11]).total_seconds())) # abs_date
    fs.append(br1-br2) # id
    fs.append(abs(br1-br2)) # abs_id
    fs.append(b1[14]-b2[14]) # crypto
    fs.append(b1[15]-b2[15]) # general
    fs.append(b1[16]-b2[16]) # java
    fs.append(b1[17]-b2[17]) # network
    fs.append(abs(b1[14]-b2[14])) # abs_crypto
    fs.append(abs(b1[15]-b2[15])) # abs_general
    fs.append(abs(b1[16]-b2[16])) # abs_java
    fs.append(abs(b1[17]-b2[17])) # abs_network
    fs.append(cos_sim(b1[14:18],b2[14:18])) #cos_sim
    ts1 = b1[-2] + ' ' + b1[-1]
    ts2 = b1[-2] + ' ' + b1[-1]
    fs.append(make_str_features(ts1, ts2))
    brt1 = brt[br1]['T']+brt[br1]['D'] if br1 in brt else []
    brt2 = brt[br2]['T']+brt[br2]['D'] if br2 in brt else []
    fs.append(make_str_features(rmv(ts1,brt1), rmv(ts2,brt2)))
    fs.append(make_str_features(rep(ts1,brt1,cts), rep(ts2,brt2,cts)))
    return fs

if __name__ == "__main__":
    cts = readFileCorrectedTerms('shCorrectedTerms.csv')
    if len(sys.argv)>=3:
        Datasets = sys.argv[2].split('-')
    else:
        Datasets = ["Android","Eclipse","Mozilla","Openoffice"]
##    Datasets = ["Android"]
    for d in Datasets:
        dps = readFileIdLbl(d+'IdLbl.txt') #duplicate pairs
        brt = readFileTypos(d+'Typos.txt',cts) # bug report with typo
        brfs = readFileFields(d+'Fields.csv') # all bug report fields
        brts = set(brt.keys())
        fdps = {}
        iii = int(sys.argv[1])
        f = open(d+'-out-'+('%2d'%(iii,))+'.txt','w')        
        f.write('j, Bug1, Bug2, class, prod, comp, type, priorA, priorS, versA, versS, date, abs_date, id, abs_id, crypto, general,java,network, abs_crypto, abs_general,abs_java,abs_network, cosine_similarity,'+
            'simBBnf1, simBBnf2, simBBnf3, simBBcm1, simBBcm2, simBBsm1, simBBsm2, simBBnml1, simBBnml2, simBBwsml, simBBwwml, simBBdsnl, simBBdswl, simBBwdsnl, simBBwdswl, simBBrld, '+
            'rmvBBnf1, rmvBBnf2, rmvBBnf3, rmvBBcm1, rmvBBcm2, rmvBBsm1, rmvBBsm2, rmvBBnml1, rmvBBnml2, rmvBBwsml, rmvBBwwml, rmvBBdsnl, rmvBBdswl, rmvBBwdsnl, rmvBBwdswl, rmvBBrld, '+
            'repBBnf1, repBBnf2, repBBnf3, repBBcm1, repBBcm2, repBBsm1, repBBsm2, repBBnml1, repBBnml2, repBBwsml, repBBwwml, repBBdsnl, repBBdswl, repBBwdsnl, repBBwdswl, repBBrld\n'
##            'simBBnf1, simBBnf2, simBBnf3, simBBcm1, simBBcm2, simBBsm1, simBBsm2, simBBnm1, simBBnm2, simBBwsm, simBBwwm, simBBdsn, simBBdsw, simBBwdsn, simBBwdsw, simBBrld, simBBrldl, '+
##            'rmvBBnf1, rmvBBnf2, rmvBBnf3, rmvBBcm1, rmvBBcm2, rmvBBsm1, rmvBBsm2, rmvBBnm1, rmvBBnm2, rmvBBwsm, rmvBBwwm, rmvBBdsn, rmvBBdsw, rmvBBwdsn, rmvBBwdsw, rmvBBrld, rmvBBrldl, '+
##            'repBBnf1, repBBnf2, repBBnf3, repBBcm1, repBBcm2, repBBsm1, repBBsm2, repBBnm1, repBBnm2, repBBwsm, repBBwwm, repBBdsn, repBBdsw, repBBwdsn, repBBwdsw, repBBrld, repBBrldl, '+
            )
        lbrtdps = list(set(brt.keys()) & set(dps.keys()) & set(brfs.keys()))
        llbrtdps = len(lbrtdps)
        rng = 50
        for j in range(iii*rng, min((iii+1)*rng,llbrtdps)):
            br = lbrtdps[j]
            dbrs =  list((set(brt.keys())-set(dps[br])-{br})&set(brfs.keys()))
            print(arrToStr([d,j,llbrtdps,br,len(dps[br]),datetime.datetime.now()]))
            for di in dps[br]:
                f.write(str(j)+',')
                f.write(arrToStr(calcAllFeat(brfs,brt,cts,br,di,'1'))[:-1])
                f.write('\n')
                f.flush()
            shuffle(dbrs)
            for di in range(0,min(max(5,3*len(dps[br])),len(dbrs))):
                f.write(str(j)+',')
                f.write(arrToStr(calcAllFeat(brfs,brt,cts,br,dbrs[di],'0'))[:-1])
                f.write('\n')
                f.flush()
        f.close()
