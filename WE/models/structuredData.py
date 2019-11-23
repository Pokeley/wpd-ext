
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchcrf import CRF

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.Wg1 = torch.zeros(6, 100 ,requires_grad=True)
        self.Wg2 = torch.zeros(6, 100, requires_grad=True)

        self.bias = torch.zeros(1, requires_grad=True)

    def forward(self, h): # h = [B, 6, 100]

        h = h.reshape(-1, 100)
        result = torch.mm(h, self.Wg1.t() ) + torch.mm(h, self.Wg2.t()) + self.bias
        result = torch.tanh( result )
        result = result.view(-1, 6, 6)
        return result

class model1(nn.Module):

    def __init__(self):
        super(model1, self).__init__()

        ### DIMENSIONS
        self.embedding_dim = 300
        #max_length = 4          # TODO change
        self.hidden_dim = 100

        num_tags = 3

        ### LAYERS
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size = self.hidden_dim // 2, # its bidirectional.
                            num_layers = 1,
                         #   dropout = 0.1,
                            bidirectional = True
                            )
        self.lstm_hidden = self.init_hidden()

        ### customized attn model.
        self.g = G()
        self.att = nn.Linear(6, 6)


        ###
        self.tmp = nn.Linear
        self.linear = nn.Linear(self.hidden_dim, 3)
        self.crfmodel = CRF( num_tags )


    def init_hidden(self):
        return ( torch.zeros(2, 1, self.hidden_dim // 2),
                 torch.zeros(2, 1, self.hidden_dim // 2) )

    def attention_net(self, lstm_output):

        # TODO maxseqsize = 6
        lstm_output = lstm_output.transpose(0, 1)               # [B, S, H]
        g = self.g(lstm_output)                                 # [B, 6, 6]

        alpha = torch.sigmoid(self.att(g))                      # [B, 6, 6]


        l = torch.zeros( alpha.size()[0], 6, self.hidden_dim, requires_grad=False)
        for idx in range(6):
            tmp = alpha[:, :,idx].unsqueeze(2) * lstm_output
            l[:, idx] = torch.sum(tmp, dim = 1)

        return l

    def forward(self, sequence):

        SEQSIZE = sequence.size()[0]
        BATCH_SIZE = sequence.size()[1]

    # LSTM
        output , _= self.lstm(sequence) # ( maxS=6, B=1, H )

    # nonlinearity
        output = torch.relu(output)

    # Attention
        att_out = self.attention_net(output) # [S, B, H]


    # nonlinearity
        att_out = torch.relu(att_out)

    # linear layer
        output = self.linear(att_out)

    # apply softmax
        output = F.log_softmax(output, dim=2)

        return output