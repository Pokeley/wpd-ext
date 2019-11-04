
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
        self.model = model_1_1031

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

        return torch.randint(0, 1, (1, 1, 300), device=DEVICE)

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

    def trainAndExtract(self, x_doms, y_pairs_label):

        batch_size = 1
        B = 1

        ### Generating input / output sequence for subproblems.
        #nodeSeq, labelSeq = sequenceGenerator(x_doms, y_pairs_label)
        x_seq, y_seq = self.structuredData(x_doms, y_pairs_label)

        assert len(x_seq) == len(y_seq)

        ### -- Embed the input, Encode the output(one-hot encode)
        for idx in range(len(x_seq)):
            data = x_seq[idx]
            label = y_seq[idx]

            embed = [ self.getPhraseEmbedding(x).view(1, 1, -1) for x in data ]
                # [ (1, 1, 300), ]

            if len(embed) == 0:
                continue
                


            

        # TODO batch 단위가 무엇인가?
        # 일단 한 사이트 내에서 batch 단위로 묶어서 가야됨...
        # optimize를 이 안에서 돌리고, 결과는 보고만 할것

        
            ### Train.
            


        ### Evaluate, Save loss.


        ### Aggregating results of subproblems.
            # single batch 가정하고 한 html에 대해서 들어왔다고 생각해, 모두 aggregate하자.

        ### Report aggreagted results and loss of the subproblem.
        return y_pairs_label, 1


    def evaluateAndExtract(self, x_dom, y_pairs_label):

        # model.eval()
        return 1, 1

    def optimize(self, ):
        return

    def _forward(self):
        pass

class NaiveModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(NaiveModel, self).__init__()
        self.hidden_size = hidden_size

        # TODO ????
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1,1,-1)  # dimension chek
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):                                   # check, should it be outside?
        return torch.zeros(1, 1, self.hidden_size, device = DEVICE)
