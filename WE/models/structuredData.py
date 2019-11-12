
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class model1(nn.Module):

    def __init__(self):
        super(model1, self).__init__()

        ### DIMENSIONS
        embedding_dim = 300
        #max_length = 4          # TODO change
        hidden_dim = 50

        ### LAYERS
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size = hidden_dim,
                            num_layers = 1,
                         #   dropout = 0.1,
                            bidirectional = True
                            )

        self.hidden2tag = nn.Linear(hidden_dim, 3)

    def init_hidden(self):
        return torch.randn() # TODO

    def forward(self, sequence):

        SEQSIZE = sequence.size()[0]

        lstm_out, _ = self.lstm(sequence)
        tag_out = self.hidden2tag(lstm_out.view(SEQSIZE, -1))

        tag_scores = F.log_softmax(tag_out, dim = 1)

        return tag_scores

