
### extractor.py

import re
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from .tree import sequenceGenerator
from .models.structuredData import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Extractor():

    def __init__(self, model, debug = False):

        self.debug = debug
        self.modelName = model
        self.embedDict = None
    #    self.embedDict = KeyedVectors.load_word2vec_format('embed/wiki-news-300d-1M.vec')


        # MODEL
        self.model = model1()
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.1)

        print("> Text Embedding loaded.")

    def structuredData(self, x_doms, y_pairs_label):

        x_seq = []
        y_seq = []

        dom = x_doms[0]
        pairs = y_pairs_label[0]
        keys = [ " ".join(re.split("[^a-zA-Z0-9]*", kv[0])) for kv in pairs ]
        values = [ " ".join(re.split("[^a-zA-Z0-9]*", kv[1])) for kv in pairs ]

        tables = dom.xpath('//table')
        for table in tables:
            rows = table.xpath('./tr')
            for row in rows:
                x_elems = []
                y_elems = []
                elems = row.xpath('./td | ./th')
                for elem in elems:
                    x_elems.append(elem)
                    if elem.text_content() is not None:
                        st = elem.text_content()
                        st = " ".join(re.split("[^a-zA-Z0-9]*", st))
                    else:
                        st = ''

                    if st in keys:              # Key
                        y_elems.append(0)
                    elif st in values:          # Value
                        y_elems.append(1)
                    else:                       # None
                        y_elems.append(2)

                x_seq.append(x_elems)
                y_seq.append(y_elems)

        return x_seq, y_seq

    def getPhraseEmbedding(self, node):

        text = node.text_content().replace('\n','').replace('\t','').strip().strip(':').strip(';')
        words = text.split(' ')

        minlen = min( 4, len(words))
        embedsForAverage = []

        return torch.randint(0, 1, (1, 1, 300), dtype=torch.float)

        """
        for word in words:
            cleaned_word = " ".join(re.split("[^a-zA-Z0-9]*", word))
            cleaned_word = cleaned_word.strip()
            if len(cleaned_word) == 0:
                continue
            try:
                embedding = self.embedDict[cleaned_word]
            except:
                embedding = self.embedDict['UNK']

            embedsForAverage.append(torch.tensor(embedding))

        # TODO embedding ==> RuntimeError: stack expects a non-empty TensorList

        meanEmbedding = torch.mean(torch.stack(embedsForAverage)) if len(words) > 0 \
                                                                  else self.embedDict['NONE']

        
        return meanEmbedding
        """
    def one_hot(self, x, class_cnt):
        return torch.eye(class_cnt)[x,:]

    def trainAndExtract(self, x_doms, y_pairs_label):

        batch_size = 1
        B = 1
        NUM_CLASS = 3

        ### Generating input / output sequence for subproblems.
        #nodeSeq, labelSeq = sequenceGenerator(x_doms, y_pairs_label)
        x_seq, y_seq = self.structuredData(x_doms, y_pairs_label)

        assert len(x_seq) == len(y_seq)

        ### -- 한 site에 대해 돌리는 것임.

        loss_accum = 0.0
        for idx in range(len(x_seq)):
            data = x_seq[idx]
            label = y_seq[idx]

            embed = [ self.getPhraseEmbedding(x).view(1, 1, -1) for x in data ]
                # [ (1, 1, 300), ]

            if len(embed) == 0:
                continue

            sequence = torch.cat(embed, 0)
            label = torch.tensor(label, dtype= torch.long )
            # (seqsize * 3)

            # TODO 수 ~ 금요일.
            # parameter 저장하는 기능 꼭 만들기.
            # 실제 string extract하고 실제 데이터랑 비교하는 부분 돌리기.
            # orf / attention layer 올리기
            # sequence 묶어서 한 batch로 만드는 기능 : pckedsequence 사용
            # 프로그램 실행 시 어떤 파일에 대해 parameter 결과 보여주고 extraction 해 주는 기능 구현.
            # 결과 visualizer attach하기. ( nodeid 붙은 놈으로 하기 / 이 때는 xpath도 같이 추출할 수 있어야 함)

            ### Train.
            self.model.zero_grad()

            scores = self.model(sequence)

            loss = self.loss_function(scores, label)
            loss.backward()

            self.optimizer.step()

            loss_accum += loss.item()

        ### Evaluate, Save loss.

        modelAcc = loss_accum / len(x_seq) if len(x_seq) is not 0 else 0;

        return y_pairs_label, modelAcc


    def evaluateAndExtract(self, x_dom, y_pairs_label):

        # model.eval()
        return 1, 1

    def optimize(self, ):
        return

    def _forward(self):
        pass

