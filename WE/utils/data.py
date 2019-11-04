
### data.py

import json
import random
import math
from lxml import etree, html

# Split data regardless of domain.
def splitData(dataList:list, trainRatio, debug = False) -> tuple:

    random.shuffle(dataList)
    trainSize = int(trainRatio * len(dataList))

    return (dataList[:trainSize] , dataList[trainSize:]) # return (trainSet, testSet)

# Split data considering domain split.
def splitDataWithDomain(dataList:list, trainRatio, debug = False) -> tuple:

    # Split by Domain
    dataByDomain = dict()

    for dat in dataList :
        key = dat['domain']
        if key in dataByDomain.keys():
            dataByDomain[key].append(dat)
        else:
            dataByDomain[key] = [dat, ]

    if debug:
        print("> Total {} domain(s) found".format(len(dataByDomain.keys())))


    lenPerDomain = []
    for k in dataByDomain.keys():
        lenPerDomain.append( (k, len(dataByDomain[k]) ) )

    trainDomain, testDomain = splitDomains( lenPerDomain, trainRatio, debug)
    trainSet = []
    testSet = []

    for td in trainDomain:
        for d in dataByDomain[td[0]]:
            trainSet.append(d)
    for td in testDomain:
        for d in dataByDomain[td[0]]:
            testSet.append(d)

    # TODO remove this !!!!!!!
    random.shuffle(trainSet)
    random.shuffle(testSet)

    return (trainSet, testSet)

def splitDomains(lenPerDomain:list, trainRatio, debug):

    totalLength = sum([a[1] for a in lenPerDomain])

    availableSet = []
    trError = 0.02 # TODO parameterize
    trmin = trainRatio - trError;
    trmax = trainRatio + trError;


    subset = _subset(lenPerDomain)
    for sub in subset :
        su = sum( (a[1] for a in sub) )
        rat = su / totalLength
        if rat >= trmin and rat <= trmax :
            availableSet.append(sub)

    if len(availableSet) == 0:
        raise ValueError("Cannot find domain split for given parameters!, Change margin or ratio.")

    selected = random.choice(availableSet)
    trainDomain = []
    testDomain = []
    realTrainSize = 0

    for elem in lenPerDomain:
        if elem in selected:
            trainDomain.append(elem)
            realTrainSize+=elem[1]
        else :
            testDomain.append(elem)

    optTrainRatio = realTrainSize / totalLength

    if debug:
        print("> Target train_ratio {}".format(trainRatio) )
        print("> By Allowing margin +/-{}, Actual train_ratio {}".format(trError, optTrainRatio))

    return (trainDomain, testDomain)

def _subset(s):
    sets = []
    for i in range(1 << len(s)):
        subset = [s[bit] for bit in range(len(s)) if _is_bit_set(i, bit)]
        sets.append(subset)
    return sets

def _is_bit_set(num, bit):
    return num & (1 << bit) > 0


class DataLoader():

    def __init__(self, debug=False):
        self.debug = debug;

    def loadfromFile(self, fileList: list, trainRatio:float = 0.7, splitDomain: bool=True):
        if len(fileList) == 0:
            raise FileNotFoundError

        if self.debug:
            print("> Total {} dataset file(s) found.".format(len(fileList)))

        # Collect data
        rawDataList = []

        for file in fileList:
            fp = open(file, 'r', encoding='utf-8')
            dataList = json.load(fp)
            for dat in dataList:
                rawDataList.append(dat)
            fp.close()

        return self._load(rawDataList, trainRatio, splitDomain)

    def loadfromJSON(self, json, trainRatio:float = 0.7, splitDomain:bool = True):
        return self._load(json, trainRatio, splitDomain)

    def _load(
            self,
            rawDataList: list,
            trainRatio: float = 0.7,
            splitDomain: bool = True
    ):

        if self.debug:
            print("> Total {} data(s) found".format(len(rawDataList)))

        res = None
        if splitDomain:
            res = splitDataWithDomain(rawDataList, trainRatio, self.debug)
        else:
            res =  splitData(rawDataList, trainRatio, self.debug)

        if self.debug:
            print("> Train data(s) {}, Test data(s) {}".format(len(res[0]), len(res[1])))

        return res

    def extractInputHTML(self, dataset):
        out = []
        for data in dataset:
            out.append(data['node'])
        return out

    def extractOutputAVPair(self, dataset):
        out = []
        for data in dataset:
            attrs = data['element_attrs']
            akeys = attrs.keys()
            datlist = []
            for key in akeys:
                elem = attrs[key]
                datlist.append ( (elem['attr'], elem['value']) )
            out.append(datlist)
        return out

    def loadDOMTree(self, data, basepath = '', addFormatstr = True):
        out = []
        for dat in data:
            filestr = basepath + dat + ('.html' if addFormatstr else '')
            fp = open(filestr, 'r')
            file = html.fromstring(fp.read())
            fp.close()
            out.append(file)
        return out

    def batchifyData(self, *datas, batch_size = 1):

        out = [ [] for _ in range(len(datas)) ]

        data_len = len(datas[0])
        # if batch_size = -1, return full-batch.
        if batch_size == -1:
            batch_size = data_len

        numBatch = math.ceil(data_len / batch_size)

        for idx in range(numBatch):

            stidx = batch_size*idx
            enidx = batch_size*(idx+1) if idx < numBatch - 1 else data_len

            for idxx, data in enumerate(datas):
                out[idxx].append(data[stidx:enidx])

        return out

class ResultExporter():

    def __init__(self, debug = False):
        self.debug = debug

    def export(self, result, exportPath, filename, type='json'):

        fileformat = type if type else '.json'
        fullpath = exportPath+filename+fileformat


        pairs = []
        for res in result:
            indict = dict()
            indict['attr'] = res[0]
            indict['value'] = res[1]
            indict['attr_xpath'] = res[2]
            indict['value_xpath'] = res[3]
            pairs.append(indict)

        # TODO file exists check
        fp = open(fullpath, 'w')
        json.dump(pairs, fp)
        fp.close()

        return