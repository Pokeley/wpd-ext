
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class model_1_1031(nn.Module):

    def __init__(self, ):
        super(model_1_1031, self).__init__()

        ### DIMENSIONS
        embedding_dim = 300
        max_length = 4          # TODO change
        hid_1_dim = 100

        ### LAYERS
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size = hid_1_dim,
                            num_layers = 1,
                            dropout = 0.1,
                            bidirectional = True
                            )


    def forward(self, x):
        pass



