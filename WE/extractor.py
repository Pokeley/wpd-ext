
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
print('DEVICE={}'.format(DEVICE))



class Extractor():

    def __init__(self, model, debug = False):

        self.FIRST_BACKWARD = True

        self.debug = debug
        self.modelName = model

        self.embedDict = None
        self.noEmbed = False
        if not self.noEmbed:
            self.embedDict = KeyedVectors.load_word2vec_format('embed/wiki-news-300d-1M.vec')

        # MODEL
        self.model = model1()
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1)

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

                    if st != '' and st in keys:              # Key
                        y_elems.append(0)
                    elif st != '' and st in values:          # Value
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

        # test.
        if self.noEmbed:
            return torch.randint(0, 1, (1, 1, 300), dtype=torch.float)


        for word in words:
            cleaned_word = " ".join(re.split("[^a-zA-Z0-9]*", word))
            cleaned_word = cleaned_word.strip()
            if len(cleaned_word) == 0:
                continue

            embedding = None
            try:
                embedding = self.embedDict[cleaned_word]
            except:
                embedding = self.embedDict['UNK']


            embedsForAverage.append(torch.tensor(embedding))

        meanEmbed = None
        if len(embedsForAverage) > 0:
            meanEmbed =  torch.mean(torch.stack(embedsForAverage), 0)
        else:
            meanEmbed =  torch.tensor(self.embedDict['NONE'])

        # print(meanEmbed.size())
        return meanEmbed

    def one_hot(self, x, class_cnt):
        return torch.eye(class_cnt)[x,:]

    def trainAndExtract(self, x_doms, y_pairs_label, backprop = True):

        ### Generating input / output sequence for subproblems.
        #nodeSeq, labelSeq = sequenceGenerator(x_doms, y_pairs_label)

        # now its only <table> only.
        x_seq, y_seq = self.structuredData(x_doms, y_pairs_label)

        assert len(x_seq) == len(y_seq)

        ### -- 한 site에 대해 돌리는 것임.

        modelError = 0
        seq_size = []
        label_list = []
        seq_list = []
        data_list = []

        # pad and batchify sequences.
        for idx in range(len(x_seq)):
            data = x_seq[idx]
            label = y_seq[idx]

            #print(data, label)

            embed = [ self.getPhraseEmbedding(x).view(1, -1) for x in data ]
                # [ (1, 1, 300), (1, 1, 300), ...... ]

            if len(embed) == 0:
                continue

            sequence = torch.cat(embed, 0) # (L, D)
            label = torch.tensor(label, dtype= torch.long ) # (L, )
            data_list.append(data)
            seq_list.append(sequence)
            seq_size.append(label.size()[0])
            label_list.append(label)

        batch_size = len(seq_list)
        if batch_size > 0:

            # TODO maxseqsize = 6
            seq_list.append( torch.zeros( (6, 300) ))

            # maxL fixed to 6.
            batched_seq = nn.utils.rnn.pad_sequence(seq_list) # (maxL, B+1, D)
            batched_seq = batched_seq[:, :-1, :]              # (maxL, B, D)

            ### Train.
            self.optimizer.zero_grad()

            output = self.model(batched_seq).view(-1, batch_size, 3)

            # Loss

            agg_loss = None
            for idx in range(batch_size):
                loss = self.loss_function(output[:seq_size[idx],idx,:], label_list[idx])

                if not agg_loss:
                    agg_loss = loss
                else:
                    if 2 not in label_list[idx].tolist():
                        agg_loss += loss
                    else:
                        agg_loss += loss / 100


            agg_loss /= batch_size
            modelError = agg_loss.item()

            if backprop:
                #agg_loss += 1.0-rate
                agg_loss.backward()

                self.optimizer.step()

            """ Extract based on result. """

            dotrain = True
            ### Extract here. ###
            extractedData = []

            right = 0
            wrong = 0
            for idx in range(batch_size):
                prediction = torch.argmax( output[:seq_size[idx],idx,:], dim = 1).tolist()
                if 1 in label_list[idx].tolist():
                    pass
                    #print(torch.tensor(prediction) , label_list[idx])

                if torch.equal(torch.tensor(prediction), label_list[idx]):
                    right+=1
                else :
                    wrong += 1

                # Extract real 'data' for real.
                seq = label_list[idx].tolist()
                seqlen = len(seq)

            #    print(len(label_list), len(seq_list), len(x_seq), batch_size)

                if seqlen == 2:
                    if (seq[0] == 0 and seq[1] == 1) and (prediction[0] == 0 and prediction[1] == 1) :
                        extractedData.append(
                                                (data_list[idx][0].text_content().replace('\n','').replace('\t','').strip().strip(':').strip(';'), \
                                                data_list[idx][1].text_content().replace('\n', '').replace('\t', '').strip().strip(':').strip(';') )

                                            )
                elif seqlen == 4:
                    if (seq[0] == 0 and seq[1] == 1) and (prediction[0] == 0 and prediction[1] == 1):

                        extractedData.append(
                            (data_list[idx][0].text_content().replace('\n', '').replace('\t', '').strip().strip(':').strip(
                                ';'), \
                             data_list[idx][1].text_content().replace('\n', '').replace('\t', '').strip().strip(':').strip(
                                 ';'))

                        )

                    if (seq[2] == 0 and seq[3] == 1) and (prediction[2] == 0 and prediction[3] == 1):

                        extractedData.append(
                            (data_list[idx][2].text_content().replace('\n', '').replace('\t', '').strip().strip(':').strip(
                                ';'), \
                             data_list[idx][3].text_content().replace('\n', '').replace('\t', '').strip().strip(':').strip(
                                 ';'))

                        )

                # Aggregate Extractexd data.
            #print(extractedData)

            rate = (right) / (right+wrong)

            extractedData = [extractedData, ]

            return extractedData , modelError, rate

        else: # case : nothing to extract.
            return [[]], -1, 0

    def evaluateAndExtract(self, x_dom, y_pairs_label):
        return self.trainAndExtract(x_dom, y_pairs_label, backprop=False)

    def optimize(self, ):
        return

    def _forward(self):
        pass