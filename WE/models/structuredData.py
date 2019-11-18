
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
        hidden_dim = 100

        ### LAYERS
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size = hidden_dim // 2, # its bidirectional.
                            num_layers = 1,
                         #   dropout = 0.1,
                            bidirectional = True
                            )

        self.linear = nn.Linear(hidden_dim, 3)


    def init_hidden(self):
        return torch.randn() # TODO

    def forward(self, sequence):

        SEQSIZE = sequence.size()[0]
        BATCH_SIZE = sequence.size()[1]

        output , _ = self.lstm(sequence) # ( maxS, B, 100 )
        output = self.linear(output)

        output = F.log_softmax(output, dim=2)

        return output # (maxS, B, 3)