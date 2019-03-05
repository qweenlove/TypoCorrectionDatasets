import gc
import time
import threading
import heapq
dicFile = 'TotalEnglishComputerTerms2.txt'
##print('Sleep 1 sec = ',GetRunTime(time.sleep,(1,)))
def GetRunTime(f,p):
    st = time.clock()
    r = f(*p)
    et = time.clock()
    return (r,(et-st)*1000) #runtime in mili second

def ReadWordsFromFile(file):
    with open(file,'r') as f:
        words = f.readlines()
        words = [w.strip() for w in words]
        words = [w for w in words if w!=""]
    return words
def MakeNeuralMatchTree(words):
##if 1==1:
##    words = ['hello','book','help','his','hiss','she']
    nmt = [{},False,None]
    for word in words:
        place = nmt
        for c in word.lower().strip():
            if c not in place[0]:
                place[0][c]=[{},False] #,place]
            place = place[0][c]
        place[1]=True
    return nmt

def CheckIsWordInNMT(nmt,word):
    place = nmt
    for c in word:
        if c in place[0]:
            place = place[0][c]
        else:
            return False
    return place[1]

def CheckIsWordInNMT2(word):
    global nmt
    return CheckIsWordInNMT(nmt,word)

def GetWordsInNMT2(word):
    global nmt
    return GetWordsInNMT(nmt,word)

def GetWordsInNMT(nmt,term):
    i = 0
    lt = len(term)
    combinations = []
    while i<lt:
        k=i
        place = nmt
        while k<=lt:
            if place[1]:
                if len(combinations)>0:
                    lb = combinations[-1][1]+1
                else:
                    lb = 0
                if lb<i:
                    combinations.append([lb,i,term[lb:i]])
                if len(combinations)>0 and combinations[-1][0]==i and combinations[-1][1]<k-1:
##                    combinations[-1]=[i,k-1,term[i:k]]
                    combinations.append([i,k-1,term[i:k]])
                else:
                    combinations.append([i,k-1,term[i:k]])
            if k==lt:
                combinations.append([i,k-1,term[i:k]])
                break
##            if term[k].isdigit() and len(combinations)==0:
##                pass
##            else:
            if term[k] not in place[0]:
                break
            place = place[0][term[k]]
            k+=1
        i+=1
    if len(combinations)>0:
        lb = combinations[-1][1]+1
    else:
        lb = 0
    if lb<lt:
        combinations.append([lb,lt,term[lb:]])
    return combinations

def GetWordsBrouteForce(nmt,term):
    combinations = []
    lt = len(term)
    for start in range(lt-1):
        for end in range(start+1,lt):
            word = term[start:end]
            if CheckIsWordInNMT(nmt,word):
                if len(combinations)>0:
                    lb = combinations[-1][1]+1
                else:
                    lb = 0
                if lb<start:
                    combinations.append((lb,start,term[lb:start]))
                if len(combinations)>0 and combinations[-1][0]==start and combinations[-1][1]<end-1:
                    combinations[-1]=[start,end-1,word]
                else:
                    combinations.append([start,end-1,word])
    if len(combinations)>0:
        lb = combinations[-1][1]+1
    else:
        lb = 0
    if lb<lt:
        combinations.append((lb,lt,term[lb:]))
    return combinations

def fff2(p,l):
    for t in p:
        if CheckIsWordInNMT2(t[2]):
            l.append(t)
            
def GetWordsBrouteForceParallel(nmt,term):
    combinations = []
    inps = [(s,e-1,term[s:e]) for s in range(len(term)-1) for e in range(s+1,len(term))]
    now = 2
    li = len(inps)//now
    threads = [threading.Thread(target=fff2,args=(inps[i*li:(i+1)*li],combinations)) for i in range(now)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    for thread in threads: del thread
    return combinations

nmt = MakeNeuralMatchTree(ReadWordsFromFile(dicFile))
##term = 'abouttoperformchange'
##combs = GetWordsInNMT(nmt,term)
def CalcAvgWordLen(combs,sel):
    s,c = 0,0
    for i in sel:
        s+=len(combs[i][2])
        c+=1
    return s/c if c>0 else 0
def ShowWordsComb(combs,sel,sep=' '):
    s = ""
    for i in sel:
        s+=combs[i][2]+sep
    return s
def GetWordsCombRec(length,combs,i=-1,sel=[]):
##    if i>=len(combs):
##        return 1,sel
    if i==-1:
        s,e=-1,-1
    else:
        if i<0 or i>=len(combs):
            return 2,[],None
        c = combs[i]
        s,e = c[0],c[1]
        if e==length-1:
##            print('****    ****   ',i,s,e)
            awl = CalcAvgWordLen(combs,sel)
            return 1,sel,awl
##    print(i,s,e)
    rs = []
    for n in [j for j in range(len(combs)) if combs[j][0]==e+1]:
        t,r,awl = GetWordsCombRec(length,combs,n,sel+[n])
        if len(r)!=0:
            if t==1:
                if awl>=2.0:
                    rs.append((r,awl,ShowWordsComb(combs,r)))
            else:
                try:
                    rs.extend(r)
                except MemoryError as err:
                    gc.collect()
    if i==-1:
##        rs = [(r,CalcAvgWordLen(combs,r),ShowWordsComb(combs,r)) for r in rs]
        rs.sort(key=lambda x: x[1],reverse=True)
        return rs
    else:
        return 2,rs,None
    
def GetWordsCombIter(length,combs,N=1000):
    i=0
    rs = []
##    states = [[-1]]
    states = [(0,[-1])]
    e = -1
    Continue = True
    while len(states)>0 and Continue:
##        r = states.pop()
        rank,r = heapq.heappop(states)
        i = r[-1]
        if i==-1:
            r = []
            e = -1
        else:
            e = combs[i][1]
        nexts = [j for j in range(len(combs)) if combs[j][0]==e+1]
##        nexts.sort(key=lambda x: (len(combs[x][2]),length-combs[x][0]), reverse=True)
##        print('e,nexts',e,[(x,combs[x][2],len(combs[x][2]),length-combs[x][0]) for x in nexts])
##        time.sleep(1)
        if len(nexts)>0:
            nexts.reverse()
        for n in nexts:
            c = combs[n]
            e = c[1]
            sel = r+[n]
            awl = CalcAvgWordLen(combs,sel)
            if e==length-1:
##                print('awl,text,sel',awl,ShowWordsComb(combs,sel),sel)
##                time.sleep(1)
                if awl>=2.0:
##                    print('**** awl,text,sel',awl,ShowWordsComb(combs,sel),sel)
                    rs.append((sel,awl,ShowWordsComb(combs,sel)))
                    if len(rs)>=N:
                        Continue = False
                        break
            else:
                heapq.heappush(states, (length-awl,sel))
##                states.append(sel)
##        del r
    rs.sort(key=lambda x: x[1],reverse=True)
    return rs

##r = GetWordsComb(len(term),combs)
##r,chkTime = GetRunTime(GetWordsComb,(len(term),combs))
##print(len(r))
def arrToStr(a, sep = ',', end='\n'):
    x=""
    for i in a:
        x = x + str(i)+sep
    return x[:-1]+end

##def TestNMT():
if 1==1:
##    words,readFileTime = GetRunTime(ReadWordsFromFile,(dicFile,))
    words,readFileTime = ['hello','book','help','his','hiss','she'],0
    nmt,nmtTime = GetRunTime(MakeNeuralMatchTree,(words,))
    print('readFileTime=',readFileTime)
    print('words=',len(words),'\nnmtTime=',nmtTime)
    ##nmt,nmtTime = GetRunTime(MakeNeuralMatchTree,(dicFile,))
    ##print('nmtTime = ' , nmtTime,'\n')
    ##print(CheckIsWordInNMT(nmt,'hello'))
    ##print(CheckIsWordInNMT(nmt,'hell'))
    ##words = GetWordsInNMT(nmt,'hellohelphissbookhishel')
    ##chk,chkTime = GetRunTime(CheckIsWordInNMT,(nmt,'hello'))
    ##print(chk,chkTime)
    term = 'hellol654shelphissbookhishel'
##    term = 'hello654helphisbookhishel'
    print(term, len(term))
    words2,w2Time = GetRunTime(GetWordsBrouteForce,(nmt,term))
    print('GetWordsBrouteForce=',len (words2),w2Time,words2[:20],'\n')
    words3,w2Time = GetRunTime(GetWordsBrouteForceParallel,(nmt,term))
    print('GetWordsBrouteForceParallel=',len(words3),w2Time,'\n')
    words4,wTime = GetRunTime(GetWordsInNMT,(nmt,term))
    print('GetWordsInNMT=',len(words4),wTime,words4[:20],'\n')
    rs1,wordsCombTime1 = GetRunTime(GetWordsCombIter,(len(term),words4,10))
    print('GetWordsCombIter=',len(rs1),wordsCombTime1,rs1)
    rs2,wordsCombTime2 = GetRunTime(GetWordsCombRec,(len(term),words4))
    print('GetWordsCombRec=',len(rs2),wordsCombTime2,rs2[:20])

if 1==2:
    term = ['abstractartifactextendsitemsemanticeditpolicy']
    combs,combsTime = GetRunTime(GetWordsInNMT2,(term[0],))
    print(term[0],len(combs))
    rs,wordsCombTime=[],-1
    rs,wordsCombTime = GetRunTime(GetWordsCombIter,(len(term[0]),combs,10))
    print(len(rs[0][0]))
    print(rs[0][1])
    print(len(combs))
    print(combsTime)
    print(len(rs))
    print(wordsCombTime)
    print(arrToStr([arrToStr([r[1],len(r[0]),r[2]],'-','') for r in rs[:10]],';',''))
    
if 1==2:
    infile = 'ConnectedSuspecious.txt'
    outfile = 'ConnectedSuspeciousResult.txt'
    with open(infile,'r') as fin:
        cs = []
        for l in fin:
            ps = l.strip().split(',')
            cs.append(ps)
    try:
        fout = open(outfile,'a')
        print('append file')
    except:
        fout = open(outfile,'w')
    term = cs[0]
    term.append('AvgWordLen')
    term.append('WordCount')
    term.append('CombsCount')
    term.append('CombsTime')
    term.append('WordsCount')
    term.append('WordsTime')
    term.append(term[4])
    del term[4]
##        term[4],term[6]=term[6],term[4]
    fout.write(arrToStr(term))
    gc.enable()
    for i in range(1,len(cs)):
        term = cs[i]
##            combs = GetWordsInNMT(nmt,term)
        combs,combsTime = GetRunTime(GetWordsInNMT2,(term[0],))
        print(term[0],len(combs))
        rs,wordsCombTime=[],-1
        try:
            rs,wordsCombTime = GetRunTime(GetWordsCombIter,(len(term[0]),combs,10))
        except MemoryError as err:
##                term[4]=len(rs[0][0])
            term.append(None)
            term.append(len(combs))
            term.append(combsTime)
            term.append(0)
            term.append(None)
            term.append('')
        except Exception as err:
            raise err
        if len(rs)>0:
            term[2]='Connected' if len(term[0])>10 and rs[0][1]>=3.0 else ''
            term[4]=len(rs[0][0])
            term.append(rs[0][1])
            term.append(len(combs))
            term.append(combsTime)
            term.append(len(rs))
            term.append(wordsCombTime)
            term.append(arrToStr([arrToStr([r[1],len(r[0]),r[2]],'-','') for r in rs[:10]],';',''))
            del rs
        sterm = arrToStr(term)
        fout.write(sterm)
        if i%20==0:
            fout.flush()
            gc.collect()
            print(sterm)
    fout.close()
                       
