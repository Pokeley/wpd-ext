
### extractor.py

import torch
from WE.tree import sequenceGenerator

class Extractor():

    def __init__(self, model, debug = False):

        self.debug = debug
        self.modelName = model
        # TODO initialize model parameters.


    def trainAndExtract(self, x_dom, y_pairs_label):


        ### Generating input / outputs for subproblems.
        sequenceGenerator(x_dom, y_pairs_label)

        ### -- Generate sequence


        ### -- Embed the input, Encode the output(one-hot encode)


        ### Train.


        ### Evaluate, Save loss.


        ### Aggregating results of subproblems.


        ### Report aggreagted results and loss of the subproblem.
        return y_pairs_label, 1


    def evaluateAndExtract(self, x_dom, y_pairs_label):

        # model.eval()
        return 1, 1

    def optimize(self, ):
        return

    def _forward(self):
        pass



